//! The conversation state machine.
//!
//! Pure: no I/O, no clock, no `AppKernel`. The host asks what to do next and
//! reports what happened, which makes every branch — approval, denial,
//! refusal, truncation, multi-tool turns — testable without a network or a
//! window.
//!
//! The loop is frame-driven rather than `async` because tool execution needs
//! main-thread access to the viewer:
//!
//! ```text
//! Idle ──user message──▶ Requesting ──turn──▶ ExecutingTools / AwaitingApproval
//!   ▲                        ▲                          │
//!   └────── no tools ────────┴───── all results in ─────┘
//! ```

use crate::approval::{classify, CommandRisk};
use crate::prompt::{user_turn_blocks, SceneSummary};
use crate::stream::AssistantTurn;
use crate::tools::{ToolCall, ToolOutcome, RUN_COMMAND};
use crate::types::{ContentBlock, ImageSource, Message, StopReason};

/// What the agent is currently doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentState {
    /// Waiting for the user.
    Idle,
    /// A request is in flight.
    Requesting,
    /// Running tools the user did not need to approve.
    ExecutingTools,
    /// Blocked on at least one approval decision.
    AwaitingApproval,
}

/// Something the host should do.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentAction {
    /// Send the current conversation to the API.
    SendRequest,
    /// Execute this tool now and report the outcome.
    ExecuteTool(ToolCall),
    /// Ask the user before executing this tool.
    RequestApproval {
        /// The call awaiting a decision.
        call: ToolCall,
        /// Why it needs one.
        risk: CommandRisk,
    },
    /// The turn finished with a normal answer.
    Completed,
    /// The turn was declined by safety classifiers.
    Refused,
}

/// A tool call the agent is tracking.
#[derive(Debug, Clone, PartialEq)]
struct PendingCall {
    call: ToolCall,
    approved: bool,
}

/// Drives one Claude conversation.
#[derive(Debug, Default)]
pub struct Agent {
    messages: Vec<Message>,
    state: AgentStateInner,
    pending: Vec<PendingCall>,
    outcomes: Vec<ToolOutcome>,
    auto_approve_mutating: bool,
}

/// Internal state, kept separate so [`AgentState`] can stay `Copy`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum AgentStateInner {
    #[default]
    Idle,
    Requesting,
    ExecutingTools,
    AwaitingApproval,
}

impl Agent {
    /// Creates an agent with the user's approval preference.
    pub fn new(auto_approve_mutating: bool) -> Self {
        Self {
            auto_approve_mutating,
            ..Default::default()
        }
    }

    /// Updates the approval preference mid-conversation.
    pub fn set_auto_approve_mutating(&mut self, value: bool) {
        self.auto_approve_mutating = value;
    }

    /// Returns the current state.
    pub fn state(&self) -> AgentState {
        match self.state {
            AgentStateInner::Idle => AgentState::Idle,
            AgentStateInner::Requesting => AgentState::Requesting,
            AgentStateInner::ExecutingTools => AgentState::ExecutingTools,
            AgentStateInner::AwaitingApproval => AgentState::AwaitingApproval,
        }
    }

    /// Returns `true` when the agent is not waiting on anything.
    pub fn is_idle(&self) -> bool {
        self.state == AgentStateInner::Idle
    }

    /// Returns the conversation so far, ready to send.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Submits a user request.
    ///
    /// Ignored unless idle, so a stray submit while a turn is in flight cannot
    /// interleave two conversations.
    pub fn submit(&mut self, summary: &SceneSummary, request: &str) -> Vec<AgentAction> {
        if self.state != AgentStateInner::Idle {
            return Vec::new();
        }
        self.messages
            .push(Message::user_blocks(user_turn_blocks(summary, request)));
        self.state = AgentStateInner::Requesting;
        vec![AgentAction::SendRequest]
    }

    /// Reports a completed assistant turn.
    pub fn turn_completed(&mut self, turn: AssistantTurn) -> Vec<AgentAction> {
        self.messages.push(Message::assistant(turn.content.clone()));

        if turn.stop_reason == Some(StopReason::Refusal) {
            self.state = AgentStateInner::Idle;
            return vec![AgentAction::Refused];
        }

        // A server-side tool loop paused mid-turn; resending continues it, and
        // the assistant turn is already on the history.
        if turn.stop_reason == Some(StopReason::PauseTurn) {
            self.state = AgentStateInner::Requesting;
            return vec![AgentAction::SendRequest];
        }

        // Trigger on the presence of tool_use blocks rather than the stop
        // reason: the blocks are what actually need answering, and an
        // unexpected stop reason should not strand them unanswered.
        let calls: Vec<ToolCall> = turn
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, name, input } => Some(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                }),
                _ => None,
            })
            .collect();

        if calls.is_empty() {
            self.state = AgentStateInner::Idle;
            return vec![AgentAction::Completed];
        }

        self.pending.clear();
        self.outcomes.clear();

        let mut actions = Vec::new();
        let mut needs_approval = false;
        for call in calls {
            let risk = self.risk_of(&call);
            if risk.requires_approval(self.auto_approve_mutating) {
                needs_approval = true;
                self.pending.push(PendingCall {
                    call: call.clone(),
                    approved: false,
                });
                actions.push(AgentAction::RequestApproval { call, risk });
            } else {
                self.pending.push(PendingCall {
                    call: call.clone(),
                    approved: true,
                });
                actions.push(AgentAction::ExecuteTool(call));
            }
        }

        self.state = if needs_approval {
            AgentStateInner::AwaitingApproval
        } else {
            AgentStateInner::ExecutingTools
        };
        actions
    }

    /// Approves a pending tool call.
    pub fn approve(&mut self, tool_use_id: &str) -> Vec<AgentAction> {
        let Some(entry) = self
            .pending
            .iter_mut()
            .find(|entry| entry.call.id == tool_use_id && !entry.approved)
        else {
            return Vec::new();
        };
        entry.approved = true;
        let call = entry.call.clone();
        self.refresh_state();
        vec![AgentAction::ExecuteTool(call)]
    }

    /// Denies a pending tool call.
    ///
    /// The denial is reported to the model as a failed tool result rather than
    /// dropped, so it can acknowledge and adapt instead of waiting forever.
    pub fn deny(&mut self, tool_use_id: &str, reason: &str) -> Vec<AgentAction> {
        let Some(entry) = self
            .pending
            .iter_mut()
            .find(|entry| entry.call.id == tool_use_id && !entry.approved)
        else {
            return Vec::new();
        };
        entry.approved = true;
        let message = if reason.trim().is_empty() {
            "The user declined to run this command.".to_string()
        } else {
            format!("The user declined to run this command: {reason}")
        };
        self.record_outcome(ToolOutcome::error(tool_use_id, message))
    }

    /// Reports the outcome of an executed tool.
    pub fn tool_finished(&mut self, outcome: ToolOutcome) -> Vec<AgentAction> {
        self.record_outcome(outcome)
    }

    /// Abandons the current turn and returns to idle.
    ///
    /// Used by the panel's stop button. Pending calls are discarded; the
    /// conversation history keeps whatever completed.
    pub fn cancel(&mut self) {
        self.pending.clear();
        self.outcomes.clear();
        self.state = AgentStateInner::Idle;
    }

    /// Records an outcome and, once every call is answered, queues the next
    /// request.
    fn record_outcome(&mut self, outcome: ToolOutcome) -> Vec<AgentAction> {
        if !self
            .pending
            .iter()
            .any(|entry| entry.call.id == outcome.tool_use_id)
        {
            // Late or duplicate outcome for a turn that already moved on.
            return Vec::new();
        }
        if self
            .outcomes
            .iter()
            .any(|existing| existing.tool_use_id == outcome.tool_use_id)
        {
            return Vec::new();
        }
        self.outcomes.push(outcome);

        if self.outcomes.len() < self.pending.len() {
            self.refresh_state();
            return Vec::new();
        }

        // Every result for the turn goes back in a single user message.
        // Splitting them across messages trains the model to stop issuing
        // parallel tool calls.
        let blocks = self
            .pending
            .iter()
            .filter_map(|entry| {
                self.outcomes
                    .iter()
                    .find(|outcome| outcome.tool_use_id == entry.call.id)
            })
            .map(outcome_to_block)
            .collect();

        self.messages.push(Message::user_blocks(blocks));
        self.pending.clear();
        self.outcomes.clear();
        self.state = AgentStateInner::Requesting;
        vec![AgentAction::SendRequest]
    }

    /// Recomputes whether anything is still awaiting a decision.
    fn refresh_state(&mut self) {
        self.state = if self.pending.iter().any(|entry| !entry.approved) {
            AgentStateInner::AwaitingApproval
        } else {
            AgentStateInner::ExecutingTools
        };
    }

    /// Classifies a call. Only command execution carries risk; the inspection
    /// tools read state and never need approval.
    fn risk_of(&self, call: &ToolCall) -> CommandRisk {
        if call.name != RUN_COMMAND {
            return CommandRisk::ReadOnly;
        }
        // A missing or non-string `command` is unusable input, so treat it the
        // same as an unrecognised command: ask.
        call.command().map_or(CommandRisk::Destructive, classify)
    }
}

/// Converts a tool outcome into the content block sent back to the model.
fn outcome_to_block(outcome: &ToolOutcome) -> ContentBlock {
    let mut content = vec![ContentBlock::text(outcome.content.clone())];
    if let Some(image) = &outcome.image_png_base64 {
        content.push(ContentBlock::Image {
            source: ImageSource::png(image.clone()),
        });
    }
    ContentBlock::ToolResult {
        tool_use_id: outcome.tool_use_id.clone(),
        content,
        is_error: outcome.is_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StopDetails, Usage};

    fn tool_use(id: &str, name: &str, input: serde_json::Value) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input,
        }
    }

    fn run_command_use(id: &str, command: &str) -> ContentBlock {
        tool_use(id, RUN_COMMAND, serde_json::json!({ "command": command }))
    }

    fn turn(content: Vec<ContentBlock>, stop_reason: StopReason) -> AssistantTurn {
        AssistantTurn {
            content,
            stop_reason: Some(stop_reason),
            stop_details: None,
            usage: Usage::default(),
            model: None,
        }
    }

    fn text_turn(text: &str) -> AssistantTurn {
        turn(vec![ContentBlock::text(text)], StopReason::EndTurn)
    }

    /// Agent that has just sent its first request.
    fn started(auto_approve: bool) -> Agent {
        let mut agent = Agent::new(auto_approve);
        let actions = agent.submit(&SceneSummary::default(), "show it as cartoon");
        assert_eq!(actions, vec![AgentAction::SendRequest]);
        agent
    }

    #[test]
    fn a_plain_answer_completes_the_turn() {
        let mut agent = started(true);

        let actions = agent.turn_completed(text_turn("Done."));

        assert_eq!(actions, vec![AgentAction::Completed]);
        assert!(agent.is_idle());
        assert_eq!(agent.messages().len(), 2);
    }

    #[test]
    fn submitting_while_busy_is_ignored() {
        // Otherwise a second submit would interleave two conversations.
        let mut agent = started(true);

        let actions = agent.submit(&SceneSummary::default(), "and now zoom");

        assert!(actions.is_empty());
        assert_eq!(agent.messages().len(), 1);
    }

    #[test]
    fn a_safe_command_executes_without_approval() {
        let mut agent = started(true);

        let actions = agent.turn_completed(turn(
            vec![run_command_use("t1", "show cartoon")],
            StopReason::ToolUse,
        ));

        assert_eq!(agent.state(), AgentState::ExecutingTools);
        assert!(matches!(actions[0], AgentAction::ExecuteTool(_)));
    }

    #[test]
    fn a_destructive_command_asks_first() {
        let mut agent = started(true);

        let actions = agent.turn_completed(turn(
            vec![run_command_use("t1", "delete all")],
            StopReason::ToolUse,
        ));

        assert_eq!(agent.state(), AgentState::AwaitingApproval);
        assert_eq!(
            actions,
            vec![AgentAction::RequestApproval {
                call: ToolCall {
                    id: "t1".to_string(),
                    name: RUN_COMMAND.to_string(),
                    input: serde_json::json!({"command": "delete all"}),
                },
                risk: CommandRisk::Destructive,
            }]
        );
    }

    #[test]
    fn mutating_commands_follow_the_auto_approve_setting() {
        let mut auto = started(true);
        let actions = auto.turn_completed(turn(
            vec![run_command_use("t1", "zoom")],
            StopReason::ToolUse,
        ));
        assert!(matches!(actions[0], AgentAction::ExecuteTool(_)));

        let mut manual = started(false);
        let actions = manual.turn_completed(turn(
            vec![run_command_use("t1", "zoom")],
            StopReason::ToolUse,
        ));
        assert!(matches!(actions[0], AgentAction::RequestApproval { .. }));
    }

    #[test]
    fn inspection_tools_never_need_approval() {
        let mut agent = started(false);

        let actions = agent.turn_completed(turn(
            vec![tool_use("t1", "list_objects", serde_json::json!({}))],
            StopReason::ToolUse,
        ));

        assert!(matches!(actions[0], AgentAction::ExecuteTool(_)));
    }

    #[test]
    fn a_command_argument_that_is_missing_or_not_a_string_asks_first() {
        for input in [serde_json::json!({}), serde_json::json!({"command": 42})] {
            let mut agent = started(true);
            let actions = agent.turn_completed(turn(
                vec![tool_use("t1", RUN_COMMAND, input.clone())],
                StopReason::ToolUse,
            ));
            assert!(
                matches!(actions[0], AgentAction::RequestApproval { .. }),
                "unusable input {input} should require approval"
            );
        }
    }

    #[test]
    fn approving_a_call_releases_it_for_execution() {
        let mut agent = started(true);
        agent.turn_completed(turn(
            vec![run_command_use("t1", "delete all")],
            StopReason::ToolUse,
        ));

        let actions = agent.approve("t1");

        assert_eq!(agent.state(), AgentState::ExecutingTools);
        assert!(matches!(actions[0], AgentAction::ExecuteTool(_)));
    }

    #[test]
    fn denying_a_call_reports_it_back_as_a_failed_tool_result() {
        // The model must learn it was refused; silence would strand the turn.
        let mut agent = started(true);
        agent.turn_completed(turn(
            vec![run_command_use("t1", "delete all")],
            StopReason::ToolUse,
        ));

        let actions = agent.deny("t1", "I still need that object");

        assert_eq!(actions, vec![AgentAction::SendRequest]);
        let last = agent.messages().last().expect("a tool result turn");
        let ContentBlock::ToolResult {
            is_error, content, ..
        } = &last.content[0]
        else {
            panic!("expected a tool result");
        };
        assert!(is_error);
        assert!(content[0]
            .as_text()
            .expect("text")
            .contains("I still need that object"));
    }

    #[test]
    fn denial_without_a_reason_still_explains_itself() {
        let mut agent = started(true);
        agent.turn_completed(turn(
            vec![run_command_use("t1", "delete all")],
            StopReason::ToolUse,
        ));

        agent.deny("t1", "   ");

        let last = agent.messages().last().expect("a tool result turn");
        let ContentBlock::ToolResult { content, .. } = &last.content[0] else {
            panic!("expected a tool result");
        };
        assert!(content[0].as_text().expect("text").contains("declined"));
    }

    #[test]
    fn all_tool_results_go_back_in_a_single_user_message() {
        // Splitting results across messages trains the model out of parallel
        // tool calls.
        let mut agent = started(true);
        agent.turn_completed(turn(
            vec![
                run_command_use("t1", "show cartoon"),
                run_command_use("t2", "zoom"),
            ],
            StopReason::ToolUse,
        ));

        assert!(agent.tool_finished(ToolOutcome::ok("t1", "ok")).is_empty());
        let actions = agent.tool_finished(ToolOutcome::ok("t2", "ok"));

        assert_eq!(actions, vec![AgentAction::SendRequest]);
        let last = agent.messages().last().expect("a tool result turn");
        assert_eq!(last.content.len(), 2, "both results in one message");
    }

    #[test]
    fn tool_results_are_ordered_to_match_the_calls() {
        let mut agent = started(true);
        agent.turn_completed(turn(
            vec![
                run_command_use("t1", "show cartoon"),
                run_command_use("t2", "zoom"),
            ],
            StopReason::ToolUse,
        ));

        // Completed out of order.
        agent.tool_finished(ToolOutcome::ok("t2", "second"));
        agent.tool_finished(ToolOutcome::ok("t1", "first"));

        let last = agent.messages().last().expect("a tool result turn");
        let ids: Vec<&str> = last
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::ToolResult { tool_use_id, .. } => tool_use_id.as_str(),
                _ => panic!("expected tool results"),
            })
            .collect();
        assert_eq!(ids, vec!["t1", "t2"]);
    }

    #[test]
    fn a_mixed_turn_waits_for_the_approval_before_sending_results() {
        let mut agent = started(true);
        let actions = agent.turn_completed(turn(
            vec![
                run_command_use("t1", "show cartoon"),
                run_command_use("t2", "delete all"),
            ],
            StopReason::ToolUse,
        ));

        assert!(matches!(actions[0], AgentAction::ExecuteTool(_)));
        assert!(matches!(actions[1], AgentAction::RequestApproval { .. }));
        assert_eq!(agent.state(), AgentState::AwaitingApproval);

        // The safe half finishing must not send a half-answered turn.
        assert!(agent.tool_finished(ToolOutcome::ok("t1", "ok")).is_empty());
        assert_eq!(agent.state(), AgentState::AwaitingApproval);

        agent.approve("t2");
        let actions = agent.tool_finished(ToolOutcome::ok("t2", "ok"));
        assert_eq!(actions, vec![AgentAction::SendRequest]);
    }

    #[test]
    fn an_image_outcome_becomes_an_image_block() {
        let mut agent = started(true);
        agent.turn_completed(turn(
            vec![tool_use("t1", "screenshot_viewport", serde_json::json!({}))],
            StopReason::ToolUse,
        ));

        agent.tool_finished(ToolOutcome::ok("t1", "Viewport capture:").with_image("QUJD"));

        let last = agent.messages().last().expect("a tool result turn");
        let ContentBlock::ToolResult { content, .. } = &last.content[0] else {
            panic!("expected a tool result");
        };
        assert_eq!(content.len(), 2);
        assert!(matches!(content[1], ContentBlock::Image { .. }));
    }

    #[test]
    fn a_refusal_ends_the_turn_without_looping() {
        let mut agent = started(true);
        let refusal = AssistantTurn {
            content: Vec::new(),
            stop_reason: Some(StopReason::Refusal),
            stop_details: Some(StopDetails {
                category: Some("bio".to_string()),
                explanation: None,
            }),
            usage: Usage::default(),
            model: None,
        };

        let actions = agent.turn_completed(refusal);

        assert_eq!(actions, vec![AgentAction::Refused]);
        assert!(agent.is_idle());
    }

    #[test]
    fn a_paused_turn_is_resumed_by_resending() {
        let mut agent = started(true);

        let actions = agent.turn_completed(turn(
            vec![ContentBlock::text("partial")],
            StopReason::PauseTurn,
        ));

        assert_eq!(actions, vec![AgentAction::SendRequest]);
        assert_eq!(agent.state(), AgentState::Requesting);
    }

    #[test]
    fn a_truncated_turn_still_completes_rather_than_hanging() {
        let mut agent = started(true);

        let actions = agent.turn_completed(turn(
            vec![ContentBlock::text("cut off mid-")],
            StopReason::MaxTokens,
        ));

        assert_eq!(actions, vec![AgentAction::Completed]);
        assert!(agent.is_idle());
    }

    #[test]
    fn tool_uses_are_answered_even_with_an_unexpected_stop_reason() {
        // Trigger on the blocks, not the stop reason: unanswered tool_use
        // blocks make the next request invalid.
        let mut agent = started(true);

        let actions = agent.turn_completed(turn(
            vec![run_command_use("t1", "zoom")],
            StopReason::Unknown("something_new".to_string()),
        ));

        assert!(matches!(actions[0], AgentAction::ExecuteTool(_)));
    }

    #[test]
    fn duplicate_and_unknown_outcomes_are_ignored() {
        let mut agent = started(true);
        agent.turn_completed(turn(
            vec![run_command_use("t1", "show cartoon")],
            StopReason::ToolUse,
        ));

        assert_eq!(
            agent.tool_finished(ToolOutcome::ok("t1", "ok")),
            vec![AgentAction::SendRequest]
        );
        // A late duplicate must not push a second, unmatched result turn.
        assert!(agent
            .tool_finished(ToolOutcome::ok("t1", "ok again"))
            .is_empty());
        assert!(agent
            .tool_finished(ToolOutcome::ok("nope", "stray"))
            .is_empty());
        assert_eq!(agent.messages().len(), 3);
    }

    #[test]
    fn approving_an_unknown_or_already_approved_call_is_a_no_op() {
        let mut agent = started(true);
        agent.turn_completed(turn(
            vec![run_command_use("t1", "delete all")],
            StopReason::ToolUse,
        ));

        assert!(!agent.approve("t1").is_empty());
        assert!(agent.approve("t1").is_empty(), "second approval is a no-op");
        assert!(agent.approve("unknown").is_empty());
    }

    #[test]
    fn cancelling_returns_to_idle_and_drops_pending_work() {
        let mut agent = started(true);
        agent.turn_completed(turn(
            vec![run_command_use("t1", "delete all")],
            StopReason::ToolUse,
        ));

        agent.cancel();

        assert!(agent.is_idle());
        // A late outcome from the abandoned turn is discarded.
        assert!(agent.tool_finished(ToolOutcome::ok("t1", "ok")).is_empty());
        // And the agent accepts new work.
        assert_eq!(
            agent.submit(&SceneSummary::default(), "try again"),
            vec![AgentAction::SendRequest]
        );
    }

    #[test]
    fn a_full_two_round_conversation_builds_valid_history() {
        let mut agent = started(true);
        agent.turn_completed(turn(
            vec![run_command_use("t1", "show cartoon")],
            StopReason::ToolUse,
        ));
        agent.tool_finished(ToolOutcome::ok("t1", "ok"));
        agent.turn_completed(text_turn("Cartoon is showing."));

        assert!(agent.is_idle());
        let roles: Vec<_> = agent.messages().iter().map(|m| m.role).collect();
        use crate::types::Role::{Assistant, User};
        assert_eq!(roles, vec![User, Assistant, User, Assistant]);
    }
}
