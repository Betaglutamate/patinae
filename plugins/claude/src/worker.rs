//! The worker thread: authentication, HTTP, and the agent loop.
//!
//! This thread never touches viewer state. When Claude asks for a tool, the
//! worker emits a [`FromWorker::ToolCall`] and blocks until the main thread
//! sends back a [`ToWorker::ToolResult`]. All viewer access happens there,
//! through `PollContext`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::api::types::{
    AssistantTurn, ContentBlock, Message, MessagesRequest, OutputConfig, StopReason, SystemBlock,
    ToolResultContent, DEFAULT_MODEL,
};
use crate::api::{self, ApiError};
use crate::auth;
use crate::settings::DEFAULT_EFFORT;
use crate::tools;

/// Ceiling on tool-call round trips within a single user turn, so a
/// pathological loop cannot run forever.
const MAX_ITERATIONS: usize = 25;

/// How long to wait on the channel before re-checking the cancel flag.
const RECV_TICK: Duration = Duration::from_millis(200);

/// Runtime configuration, snapshotted from settings on each turn.
#[derive(Debug, Clone)]
pub struct Config {
    pub model: String,
    pub effort: String,
    pub max_tokens: u32,
    pub allow_python: bool,
    /// Complete system prompt: the static skill document plus the command
    /// inventory generated from the live registry. Must stay byte-identical
    /// across turns or the prompt cache never hits.
    pub system_prompt: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            effort: DEFAULT_EFFORT.to_string(),
            max_tokens: 64_000,
            allow_python: true,
            system_prompt: String::new(),
        }
    }
}

/// Main thread → worker.
#[derive(Debug)]
pub enum ToWorker {
    /// A new user turn.
    Prompt(String),
    /// The outcome of a tool call the worker requested.
    ToolResult {
        call_id: String,
        content: Vec<ToolResultContent>,
        is_error: bool,
    },
    /// Refresh the config snapshot (settings changed, or the command inventory
    /// became available).
    Config(Box<Config>),
    /// Run the interactive `ant` sign-in.
    Login,
    /// Clear the `ant` credential.
    Logout,
    /// Re-check auth state and report it.
    RefreshAuth,
    /// Drop conversation history.
    Reset,
    Shutdown,
}

/// Worker → main thread.
#[derive(Debug, Clone)]
pub enum FromWorker {
    /// Auth/status line for the panel header.
    Status(String),
    /// Whether a turn is in flight.
    Busy(bool),
    /// Streamed assistant prose.
    TextDelta(String),
    /// An out-of-band note for the transcript (tool calls, errors, refusals).
    Note(String),
    /// Claude wants a tool executed on the main thread.
    ToolCall {
        call_id: String,
        name: String,
        input: Box<Value>,
    },
    /// The turn finished (successfully or not).
    TurnDone,
}

/// Handle held by the main thread.
pub struct WorkerHandle {
    pub tx: Sender<ToWorker>,
    pub rx: Receiver<FromWorker>,
    pub cancel: Arc<AtomicBool>,
}

impl WorkerHandle {
    /// Ask the worker to abandon the current turn at the next checkpoint.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

/// Spawn the worker thread.
pub fn spawn() -> WorkerHandle {
    let (to_tx, to_rx) = std::sync::mpsc::channel::<ToWorker>();
    let (from_tx, from_rx) = std::sync::mpsc::channel::<FromWorker>();
    let cancel = Arc::new(AtomicBool::new(false));

    let worker_cancel = Arc::clone(&cancel);
    std::thread::Builder::new()
        .name("claude-agent".to_string())
        .spawn(move || {
            let mut worker = Worker::new(to_rx, from_tx, worker_cancel);
            worker.run();
        })
        .expect("failed to spawn claude-agent thread");

    WorkerHandle {
        tx: to_tx,
        rx: from_rx,
        cancel,
    }
}

struct Worker {
    rx: Receiver<ToWorker>,
    tx: Sender<FromWorker>,
    cancel: Arc<AtomicBool>,
    config: Config,
    history: Vec<Message>,
    runtime: Option<tokio::runtime::Runtime>,
    client: Option<reqwest::Client>,
}

impl Worker {
    fn new(rx: Receiver<ToWorker>, tx: Sender<FromWorker>, cancel: Arc<AtomicBool>) -> Self {
        Self {
            rx,
            tx,
            cancel,
            config: Config::default(),
            history: Vec::new(),
            runtime: None,
            client: None,
        }
    }

    fn send(&self, msg: FromWorker) {
        let _ = self.tx.send(msg);
    }

    fn note(&self, text: impl Into<String>) {
        self.send(FromWorker::Note(text.into()));
    }

    fn run(&mut self) {
        self.report_auth();
        while let Ok(msg) = self.rx.recv() {
            match msg {
                ToWorker::Prompt(p) => self.run_turn(p),
                ToWorker::Config(c) => self.config = *c,
                ToWorker::Login => self.do_login(),
                ToWorker::Logout => self.do_logout(),
                ToWorker::RefreshAuth => self.report_auth(),
                ToWorker::Reset => {
                    self.history.clear();
                    self.note("Conversation cleared.");
                }
                ToWorker::Shutdown => break,
                // A stray result for a turn that already ended.
                ToWorker::ToolResult { .. } => {}
            }
        }
    }

    fn report_auth(&self) {
        self.send(FromWorker::Status(auth::state().describe().to_string()));
    }

    fn do_login(&self) {
        self.send(FromWorker::Status(
            "Signing in — check your browser…".to_string(),
        ));
        match auth::login() {
            Ok(output) => {
                if !output.trim().is_empty() {
                    self.note(output);
                }
                self.send(FromWorker::Status("Signed in.".to_string()));
            }
            Err(e) => {
                self.note(format!("Sign-in failed: {e}"));
                self.report_auth();
            }
        }
    }

    fn do_logout(&self) {
        if let Err(e) = auth::logout() {
            self.note(format!("Sign-out failed: {e}"));
        }
        self.report_auth();
    }

    /// Lazily build the Tokio runtime and HTTP client, so a user who never
    /// signs in pays nothing.
    fn ensure_transport(&mut self) -> Result<(), String> {
        if self.runtime.is_none() {
            self.runtime = Some(
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| format!("failed to start async runtime: {e}"))?,
            );
        }
        if self.client.is_none() {
            self.client = Some(api::build_client().map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    /// Drive one user turn to completion, including any tool round trips.
    fn run_turn(&mut self, prompt: String) {
        self.cancel.store(false, Ordering::SeqCst);
        self.send(FromWorker::Busy(true));

        if let Err(e) = self.ensure_transport() {
            self.note(e);
            self.finish_turn();
            return;
        }

        self.history.push(Message::user_text(prompt));

        for iteration in 0..MAX_ITERATIONS {
            if self.cancelled() {
                self.note("Stopped.");
                break;
            }

            let turn = match self.request() {
                Ok(t) => t,
                Err(e) => {
                    self.note(format!("Request failed: {e}"));
                    break;
                }
            };

            // Append the full content — dropping tool_use or thinking blocks
            // here would invalidate the next request.
            if !turn.content.is_empty() {
                self.history.push(Message::assistant(turn.content.clone()));
            }

            match turn.stop_reason {
                Some(StopReason::ToolUse) => match self.dispatch_tools(&turn) {
                    Some(results) => self.history.push(Message::user(results)),
                    None => break, // cancelled or shut down mid-flight
                },
                Some(StopReason::PauseTurn) => continue,
                Some(StopReason::Refusal) => {
                    let detail = turn
                        .stop_details
                        .as_ref()
                        .and_then(|d| d.explanation.clone())
                        .unwrap_or_else(|| "no explanation provided".to_string());
                    self.note(format!("Claude declined this request ({detail})."));
                    break;
                }
                Some(StopReason::MaxTokens) => {
                    self.note("Response hit the max_tokens limit and was truncated.");
                    break;
                }
                _ => break,
            }

            if iteration + 1 == MAX_ITERATIONS {
                self.note(format!(
                    "Stopped after {MAX_ITERATIONS} tool round trips. Ask again to continue."
                ));
            }
        }

        self.finish_turn();
    }

    fn finish_turn(&self) {
        self.send(FromWorker::Busy(false));
        self.send(FromWorker::TurnDone);
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    /// Issue one streaming request, retrying once if the token was rejected.
    fn request(&mut self) -> Result<AssistantTurn, String> {
        match self.request_once() {
            Err(ApiError::Unauthorized(_)) => {
                // The cached token may simply have aged out; force a refresh.
                auth::invalidate_cache();
                self.request_once().map_err(|e| e.to_string())
            }
            other => other.map_err(|e| e.to_string()),
        }
    }

    fn request_once(&mut self) -> Result<AssistantTurn, ApiError> {
        let token = auth::access_token().map_err(|e| ApiError::Unauthorized(e.to_string()))?;

        let request = MessagesRequest {
            model: self.config.model.clone(),
            max_tokens: self.config.max_tokens,
            messages: self.history.clone(),
            system: if self.config.system_prompt.is_empty() {
                vec![]
            } else {
                vec![SystemBlock::cached(&self.config.system_prompt)]
            },
            tools: tools::schemas(self.config.allow_python),
            output_config: Some(OutputConfig {
                effort: Some(self.config.effort.clone()),
            }),
            stream: true,
        };

        let client = self.client.as_ref().expect("transport initialised").clone();
        let tx = self.tx.clone();
        let runtime = self.runtime.as_ref().expect("runtime initialised");

        runtime.block_on(async move {
            api::stream_message(&client, &token, &request, |event| {
                use crate::api::stream::TurnEvent;
                let msg = match event {
                    TurnEvent::TextDelta(t) => Some(FromWorker::TextDelta(t)),
                    TurnEvent::ThinkingStarted => {
                        Some(FromWorker::Status("Claude is thinking…".to_string()))
                    }
                    TurnEvent::ToolUseStarted { name } => {
                        Some(FromWorker::Status(format!("Running {name}…")))
                    }
                    TurnEvent::Error(e) => Some(FromWorker::Note(format!("Stream error: {e}"))),
                    TurnEvent::Done => None,
                };
                if let Some(msg) = msg {
                    let _ = tx.send(msg);
                }
            })
            .await
        })
    }

    /// Emit every tool call in the turn, then wait for all their results.
    ///
    /// Returns `None` if the turn was cancelled or the channel closed — the API
    /// requires a result for every `tool_use` block, so a partial set is
    /// unusable and the turn must be abandoned instead.
    fn dispatch_tools(&mut self, turn: &AssistantTurn) -> Option<Vec<ContentBlock>> {
        let calls: Vec<(String, String)> = turn
            .tool_uses()
            .into_iter()
            .map(|(id, name, input)| {
                self.send(FromWorker::ToolCall {
                    call_id: id.to_string(),
                    name: name.to_string(),
                    input: Box::new(input.clone()),
                });
                (id.to_string(), name.to_string())
            })
            .collect();

        let mut results: Vec<ContentBlock> = Vec::with_capacity(calls.len());
        while results.len() < calls.len() {
            if self.cancelled() {
                self.note("Stopped.");
                return None;
            }
            match self.rx.recv_timeout(RECV_TICK) {
                Ok(ToWorker::ToolResult {
                    call_id,
                    content,
                    is_error,
                }) => {
                    results.push(if is_error {
                        let text = content
                            .iter()
                            .filter_map(|c| match c {
                                ToolResultContent::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        ContentBlock::tool_err(call_id, text)
                    } else {
                        ContentBlock::tool_ok(call_id, content)
                    });
                }
                // Config updates are safe to apply mid-turn; they take effect
                // on the next request.
                Ok(ToWorker::Config(c)) => self.config = *c,
                Ok(ToWorker::Shutdown) => return None,
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return None,
            }
        }
        Some(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_targets_sonnet_5_at_medium() {
        let c = Config::default();
        assert_eq!(c.model, "claude-sonnet-5");
        assert_eq!(c.effort, "medium");
        assert_eq!(c.max_tokens, 64_000);
    }

    #[test]
    fn cancel_flag_is_visible_through_the_handle() {
        let cancel = Arc::new(AtomicBool::new(false));
        let handle = WorkerHandle {
            tx: std::sync::mpsc::channel().0,
            rx: std::sync::mpsc::channel().1,
            cancel: Arc::clone(&cancel),
        };
        assert!(!cancel.load(Ordering::SeqCst));
        handle.cancel();
        assert!(cancel.load(Ordering::SeqCst));
    }

    #[test]
    fn worker_starts_and_shuts_down_cleanly() {
        let handle = spawn();
        // The worker reports auth state on startup without being asked.
        assert!(matches!(
            handle.rx.recv_timeout(Duration::from_secs(5)),
            Ok(FromWorker::Status(_))
        ));
        handle.tx.send(ToWorker::Shutdown).unwrap();
    }

    #[test]
    fn reset_clears_history_and_reports() {
        let handle = spawn();
        let _ = handle.rx.recv_timeout(Duration::from_secs(5));
        handle.tx.send(ToWorker::Reset).unwrap();
        match handle.rx.recv_timeout(Duration::from_secs(5)) {
            Ok(FromWorker::Note(n)) => assert!(n.contains("cleared")),
            other => panic!("expected a Note, got {other:?}"),
        }
        handle.tx.send(ToWorker::Shutdown).unwrap();
    }
}
