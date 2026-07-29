//! Main-thread message handler.
//!
//! Owns the bridge between the worker thread and the viewer. Everything that
//! touches viewer state happens here, inside `poll()`, because that is the only
//! place a plugin is handed a `PollContext`.
//!
//! Two execution paths, because the host offers two different mechanisms:
//!
//! - **Command-backed tools** go through `execute_command`, whose result only
//!   arrives on a *later* poll. `pending_execs` maps the host's execution id
//!   back to the Claude tool-call id so the result can be paired up.
//! - **Viewer-reading tools** run inside a `queue_viewer_mutation` closure.
//!   That call returns nothing, so each closure owns a clone of the worker
//!   sender and posts its own result.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use base64::Engine as _;
use patinae_plugin::prelude::*;
use serde_json::Value;

use crate::prompt;
use crate::provider::Part;
use crate::settings;
use crate::state::{Entry, Pending, Shared};
use crate::tools;
use crate::worker::{FromWorker, ToWorker, WorkerHandle};

/// Source of host execution ids. Unique per process, so results can never be
/// confused across plugin instances.
static NEXT_EXEC_ID: AtomicU64 = AtomicU64::new(1);

/// A tool call held back until the user approves it.
struct Gated {
    call_id: String,
    name: String,
    command: String,
}

pub struct ClaudeHandler {
    worker: WorkerHandle,
    state: Shared,
    /// host execution id -> (tool-call id, tool name)
    pending_execs: HashMap<u64, (String, String)>,
    /// Approval queue; the front entry is the one shown in the panel.
    gated: VecDeque<Gated>,
    /// Config last pushed to the worker, so settings changes are detected
    /// without resending on every frame.
    last_config: Option<String>,
    /// The command inventory is built from `ctx.registry()`, which is only
    /// reachable from a command — not from `poll()`.
    prompt_ready: bool,
}

impl ClaudeHandler {
    pub fn new(worker: WorkerHandle, state: Shared) -> Self {
        Self {
            worker,
            state,
            pending_execs: HashMap::new(),
            gated: VecDeque::new(),
            last_config: None,
            prompt_ready: false,
        }
    }

    fn send_worker(&self, msg: ToWorker) {
        let _ = self.worker.tx.send(msg);
    }

    fn note(&self, text: impl Into<String>) {
        if let Ok(mut s) = self.state.lock() {
            s.push(Entry::Note(text.into()));
        }
    }

    /// Post a tool result back to the worker.
    ///
    /// The tool name travels with the result because Gemini's
    /// `functionResponse` requires it; Anthropic pairs by id alone.
    fn reply(&self, call_id: &str, name: &str, content: Vec<Part>, is_error: bool) {
        self.send_worker(ToWorker::ToolResult {
            call_id: call_id.to_string(),
            name: name.to_string(),
            content,
            is_error,
        });
    }

    fn reply_text(&self, call_id: &str, name: &str, text: impl Into<String>, is_error: bool) {
        self.reply(call_id, name, vec![Part::text(text)], is_error);
    }

    // --- settings -------------------------------------------------------

    /// Push a fresh config to the worker whenever anything relevant changed.
    fn sync_config(&mut self, ctx: &mut PollContext<'_>) {
        // Holding back the config until the system prompt exists would leave
        // the agent with no Patinae knowledge on its very first turn.
        let system_prompt = prompt::cached().unwrap_or_default();
        if !system_prompt.is_empty() {
            self.prompt_ready = true;
        }

        let id = settings::provider(ctx.shared);
        let config = crate::worker::Config {
            provider: id,
            model: settings::model(ctx.shared, id),
            effort: settings::effort(ctx.shared),
            max_tokens: settings::max_tokens(ctx.shared),
            allow_python: settings::allow_python(ctx.shared),
            system_prompt,
        };
        let fingerprint = format!(
            "{}|{}|{}|{}|{}|{}",
            config.provider.as_str(),
            config.model,
            config.effort,
            config.max_tokens,
            config.allow_python,
            config.system_prompt.len()
        );
        if self.last_config.as_deref() != Some(fingerprint.as_str()) {
            self.last_config = Some(fingerprint);
            self.send_worker(ToWorker::Config(Box::new(config)));
        }
    }

    // --- panel-driven requests -----------------------------------------

    fn drain_panel_requests(&mut self, ctx: &mut PollContext<'_>) {
        let (submit, decision, cancel, reset, login, logout) = {
            let Ok(mut s) = self.state.lock() else {
                return;
            };
            (
                s.submit.take(),
                s.decision.take(),
                std::mem::take(&mut s.cancel_requested),
                std::mem::take(&mut s.reset_requested),
                std::mem::take(&mut s.login_requested),
                std::mem::take(&mut s.logout_requested),
            )
        };

        if let Some(prompt) = submit {
            if let Ok(mut s) = self.state.lock() {
                s.push(Entry::User(prompt.clone()));
            }
            self.send_worker(ToWorker::Prompt(prompt));
        }
        if let Some(allow) = decision {
            self.resolve_gate(ctx, allow);
        }
        if cancel {
            self.worker.cancel();
        }
        if reset {
            self.gated.clear();
            self.pending_execs.clear();
            if let Ok(mut s) = self.state.lock() {
                s.transcript.clear();
                s.pending = None;
                s.dirty = true;
            }
            self.send_worker(ToWorker::Reset);
        }
        if login {
            self.send_worker(ToWorker::Login);
        }
        if logout {
            self.send_worker(ToWorker::Logout);
        }
    }

    /// Approve or deny the front of the approval queue.
    fn resolve_gate(&mut self, ctx: &mut PollContext<'_>, allow: bool) {
        let Some(gate) = self.gated.pop_front() else {
            return;
        };

        if allow {
            self.note(format!("{} → {}", gate.name, gate.command));
            self.run_command_tool(ctx, &gate.call_id, &gate.name, &gate.command);
        } else {
            self.note(format!("Denied: {}", gate.command));
            // An explicit error result lets Claude adapt; silently dropping the
            // call would strand the turn waiting forever.
            self.reply_text(
                &gate.call_id,
                &gate.name,
                "The user denied this tool call. Do not retry it; explain what you \
                 wanted to do and ask how they would like to proceed.",
                true,
            );
        }
        self.show_next_gate();
    }

    fn show_next_gate(&self) {
        if let Ok(mut s) = self.state.lock() {
            s.pending = self.gated.front().map(|g| Pending {
                call_id: g.call_id.clone(),
                name: g.name.clone(),
                payload: g.command.clone(),
            });
            s.dirty = true;
        }
    }

    // --- worker events --------------------------------------------------

    fn drain_worker(&mut self, ctx: &mut PollContext<'_>) {
        while let Ok(msg) = self.worker.rx.try_recv() {
            match msg {
                FromWorker::Status(text) => {
                    if let Ok(mut s) = self.state.lock() {
                        s.status = text;
                        s.dirty = true;
                    }
                }
                FromWorker::Busy(busy) => {
                    if let Ok(mut s) = self.state.lock() {
                        s.busy = busy;
                        s.dirty = true;
                    }
                }
                FromWorker::TextDelta(text) => {
                    if let Ok(mut s) = self.state.lock() {
                        s.push_text_delta(&text);
                    }
                }
                FromWorker::Note(text) => self.note(text),
                FromWorker::ToolCall {
                    call_id,
                    name,
                    input,
                } => self.dispatch_tool(ctx, &call_id, &name, &input),
                FromWorker::TurnDone => {
                    if let Ok(mut s) = self.state.lock() {
                        s.busy = false;
                        s.dirty = true;
                    }
                }
            }
        }
    }

    /// Route one tool call to the right execution path.
    fn dispatch_tool(
        &mut self,
        ctx: &mut PollContext<'_>,
        call_id: &str,
        name: &str,
        input: &Value,
    ) {
        // Command-backed tools may need approval first.
        if let Some(command) = tools::command_for(name, input) {
            let auto = ctx.shared.setting_bool(settings::AUTO_APPROVE, false);
            // Python executes arbitrary code in the viewer's own process, so it
            // always prompts regardless of the auto-approve setting.
            let always_ask = name == tools::RUN_PYTHON;
            if auto && !always_ask {
                self.note(format!("{name} → {command}"));
                self.run_command_tool(ctx, call_id, name, &command);
            } else {
                self.gated.push_back(Gated {
                    call_id: call_id.to_string(),
                    name: name.to_string(),
                    command,
                });
                self.show_next_gate();
            }
            return;
        }

        if tools::is_mutating(name) {
            // A mutating tool with no command form means a malformed input.
            self.reply_text(
                call_id,
                name,
                format!("`{name}` was called without its required argument."),
                true,
            );
            return;
        }

        match name {
            tools::SCREENSHOT => self.run_screenshot(ctx, call_id, name, input),
            tools::GET_SCENE_STATE => self.run_read(ctx, call_id, name, tools::scene_state),
            tools::COUNT_ATOMS => {
                let expr = input
                    .get("selection")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                if expr.trim().is_empty() {
                    self.reply_text(call_id, name, "`count_atoms` requires a `selection`.", true);
                } else {
                    self.run_read(ctx, call_id, name, move |v| tools::count_atoms(v, &expr));
                }
            }
            tools::GET_SEQUENCES => {
                let object = input
                    .get("object")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let chain = input
                    .get("chain")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                self.run_read(ctx, call_id, name, move |v| {
                    tools::sequences(v, object.as_deref(), chain.as_deref())
                });
            }
            other => self.reply_text(call_id, other, format!("Unknown tool `{other}`."), true),
        }
    }

    /// Queue a command for the host executor and remember which tool call it
    /// belongs to; the result surfaces in a later poll.
    fn run_command_tool(
        &mut self,
        ctx: &mut PollContext<'_>,
        call_id: &str,
        name: &str,
        command: &str,
    ) {
        let exec_id = NEXT_EXEC_ID.fetch_add(1, Ordering::SeqCst);
        self.pending_execs
            .insert(exec_id, (call_id.to_string(), name.to_string()));
        ctx.execute_command(exec_id, command, true);
    }

    /// Run a read-only closure against the viewer on the main thread.
    fn run_read<F>(&self, ctx: &mut PollContext<'_>, call_id: &str, name: &str, read: F)
    where
        F: FnOnce(&dyn ViewerLike) -> String + Send + 'static,
    {
        let tx = self.worker.tx.clone();
        let call_id = call_id.to_string();
        let name = name.to_string();
        ctx.queue_viewer_mutation(move |viewer| {
            let text = read(viewer);
            let _ = tx.send(ToWorker::ToolResult {
                call_id,
                name,
                content: vec![Part::text(text)],
                is_error: false,
            });
        });
    }

    /// Capture the viewport and return it as an image block.
    ///
    /// `capture_png` only writes to a path, so this round-trips through a temp
    /// file. The closure runs with a live render context because the host
    /// supplies one when applying plugin mutations.
    fn run_screenshot(&self, ctx: &mut PollContext<'_>, call_id: &str, name: &str, input: &Value) {
        let (default_w, default_h) = settings::capture_size(ctx.shared);
        let width = input
            .get("width")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(default_w);
        let height = input
            .get("height")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(default_h);

        let tx = self.worker.tx.clone();
        let call_id = call_id.to_string();
        let name = name.to_string();
        let path = std::env::temp_dir().join(format!("patinae-claude-{call_id}.png"));

        ctx.queue_viewer_mutation(move |viewer| {
            let result = viewer.capture_png(&path, Some(width), Some(height));
            let message = match result {
                Ok(()) => match std::fs::read(&path) {
                    Ok(bytes) => {
                        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                        ToWorker::ToolResult {
                            call_id,
                            name,
                            content: vec![
                                Part::text(format!("Viewport capture, {width}x{height}:")),
                                Part::png(encoded),
                            ],
                            is_error: false,
                        }
                    }
                    Err(e) => ToWorker::ToolResult {
                        call_id,
                        name,
                        content: vec![Part::text(format!(
                            "Capture succeeded but the image could not be read back: {e}"
                        ))],
                        is_error: true,
                    },
                },
                Err(e) => ToWorker::ToolResult {
                    call_id,
                    name,
                    content: vec![Part::text(format!("Screenshot failed: {e}"))],
                    is_error: true,
                },
            };
            let _ = std::fs::remove_file(&path);
            let _ = tx.send(message);
        });
    }

    /// Pair completed host command executions back to their tool calls.
    fn drain_command_results(&mut self, ctx: &mut PollContext<'_>) {
        for result in ctx.command_results {
            let Some((call_id, name)) = self.pending_execs.remove(&result.id) else {
                continue;
            };
            match &result.result {
                Ok(()) => {
                    self.reply_text(&call_id, &name, "Command completed successfully.", false)
                }
                Err(e) => self.reply_text(&call_id, &name, format!("Command failed: {e}"), true),
            }
        }
    }
}

impl MessageHandler for ClaudeHandler {
    fn on_message(&mut self, _msg: &AppMessage, _bus: &mut MessageBus) {}

    fn needs_poll(&self) -> bool {
        true
    }

    fn poll(&mut self, ctx: &mut PollContext<'_>) {
        self.sync_config(ctx);
        self.drain_command_results(ctx);
        self.drain_panel_requests(ctx);
        self.drain_worker(ctx);

        let dirty = self
            .state
            .lock()
            .map(|mut s| std::mem::take(&mut s.dirty))
            .unwrap_or(false);
        if dirty {
            ctx.request_panel_update();
        }
    }
}
