//! Assembles OpenRouter chat-completion chunks into a completed turn.
//!
//! The OpenAI streaming dialect is the fiddliest of the three, for two reasons.
//!
//! **Tool calls stream as fragments of a JSON string, keyed by position.** The
//! first fragment for an index carries the id and function name; every later one
//! extends `arguments` with the next few characters of what will eventually be
//! valid JSON. Nothing marks the end. So calls are accumulated into slots by
//! index and only parsed once the stream closes.
//!
//! **`reasoning_details` arrive in pieces too**, and upstream vendors validate
//! them as a contiguous sequence when the conversation continues. Fragments of
//! one block therefore have to be rejoined rather than appended as separate
//! blocks, or the replayed trace is rejected.

use super::types::{AssistantTurn, ChatChunk, ToolCall, Usage};

/// Sentinel that closes an OpenAI-dialect stream. Not JSON, so it is matched
/// before any attempt to parse.
pub const DONE_SENTINEL: &str = "[DONE]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnEvent {
    TextDelta(String),
    ReasoningStarted,
    ToolUseStarted { name: String },
    Error(String),
    Done,
}

/// A tool call still being streamed.
#[derive(Debug, Default, Clone)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
pub struct TurnAccumulator {
    text: String,
    calls: Vec<Option<PartialCall>>,
    reasoning_details: Vec<serde_json::Value>,
    finish_reason: Option<String>,
    usage: Usage,
    model: String,
    reasoning_announced: bool,
}

impl TurnAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one SSE `data` payload.
    pub fn push(&mut self, data: &str) -> Vec<TurnEvent> {
        let data = data.trim();
        if data == DONE_SENTINEL {
            return vec![TurnEvent::Done];
        }

        let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
            return vec![];
        };

        // A top-level error replaces the chunk entirely; a per-choice error
        // arrives beside one. Both mean the upstream provider gave up.
        if let Some(message) = error_message(&value) {
            return vec![TurnEvent::Error(message)];
        }

        let Ok(chunk) = serde_json::from_value::<ChatChunk>(value) else {
            return vec![];
        };

        if let Some(m) = chunk.model {
            self.model = m;
        }
        if let Some(u) = chunk.usage {
            self.usage = u;
        }

        let mut events = Vec::new();
        for choice in chunk.choices {
            if let Some(err) = choice.error.as_ref().and_then(error_message) {
                events.push(TurnEvent::Error(err));
            }
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
            let Some(delta) = choice.delta else {
                continue;
            };

            if let Some(text) = delta.content {
                if !text.is_empty() {
                    self.text.push_str(&text);
                    events.push(TurnEvent::TextDelta(text));
                }
            }

            // Reasoning is a progress signal, never transcript prose — pasting
            // a model's private reasoning into its answer is exactly what the
            // separate field exists to prevent.
            let reasoning_seen =
                delta.reasoning.is_some_and(|r| !r.is_empty()) || delta.reasoning_details.is_some();
            if reasoning_seen && !self.reasoning_announced {
                self.reasoning_announced = true;
                events.push(TurnEvent::ReasoningStarted);
            }
            for detail in delta.reasoning_details.into_iter().flatten() {
                self.absorb_reasoning(detail);
            }

            for call in delta.tool_calls.into_iter().flatten() {
                if let Some(name) = self.absorb_call(call) {
                    events.push(TurnEvent::ToolUseStarted { name });
                }
            }
        }
        events
    }

    /// Fold one tool-call fragment into its slot, naming the tool the first time
    /// that slot is filled.
    fn absorb_call(&mut self, delta: super::types::ToolCallDelta) -> Option<String> {
        if self.calls.len() <= delta.index {
            self.calls.resize_with(delta.index + 1, || None);
        }
        let slot = self.calls[delta.index].get_or_insert_with(PartialCall::default);

        if let Some(id) = delta.id {
            slot.id = id;
        }
        let mut announced = None;
        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                // The name can itself be split across fragments; announce only
                // when the slot goes from unnamed to named.
                if slot.name.is_empty() && !name.is_empty() {
                    announced = Some(name.clone());
                }
                slot.name.push_str(&name);
            }
            if let Some(arguments) = function.arguments {
                slot.arguments.push_str(&arguments);
            }
        }
        announced
    }

    /// Rejoin a reasoning fragment with the block it belongs to.
    ///
    /// Fragments of one block share a `type` and an `index`; upstream vendors
    /// validate the sequence as the model emitted it, so appending each fragment
    /// as its own block would produce a trace they reject.
    fn absorb_reasoning(&mut self, detail: serde_json::Value) {
        let same_block = |a: &serde_json::Value, b: &serde_json::Value| {
            a.get("type") == b.get("type") && a.get("index") == b.get("index")
        };

        if let Some(last) = self.reasoning_details.last_mut() {
            if same_block(last, &detail) {
                // Only the prose field grows; signatures and ids arrive whole.
                for key in ["text", "summary", "data"] {
                    if let (Some(existing), Some(addition)) = (
                        last.get(key).and_then(|v| v.as_str()).map(str::to_string),
                        detail.get(key).and_then(|v| v.as_str()),
                    ) {
                        last[key] = serde_json::Value::String(format!("{existing}{addition}"));
                    }
                }
                // A signature usually lands on the closing fragment.
                for key in ["signature", "id", "format"] {
                    if let Some(v) = detail.get(key) {
                        last[key] = v.clone();
                    }
                }
                return;
            }
        }
        self.reasoning_details.push(detail);
    }

    pub fn finish(self) -> AssistantTurn {
        AssistantTurn {
            text: self.text,
            tool_calls: self
                .calls
                .into_iter()
                .flatten()
                // A slot that never received a name is an artefact of a
                // truncated stream, not a call anyone can execute.
                .filter(|c| !c.name.is_empty())
                .map(|c| ToolCall::function(c.id, c.name, c.arguments))
                .collect(),
            reasoning_details: self.reasoning_details,
            finish_reason: self.finish_reason,
            usage: self.usage,
            model: self.model,
        }
    }
}

/// Pull a human-readable message out of an error payload in either of the
/// shapes OpenRouter uses.
fn error_message(value: &serde_json::Value) -> Option<String> {
    let error = value.get("error").unwrap_or(value);
    if error.is_null() {
        return None;
    }
    match error {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(_) => error
            .get("message")
            .and_then(|m| m.as_str())
            .map(str::to_string),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::SseDecoder;

    fn run(body: &[u8]) -> (Vec<TurnEvent>, AssistantTurn) {
        let mut decoder = SseDecoder::new();
        let mut acc = TurnAccumulator::new();
        let mut events = Vec::new();
        for payload in decoder.push(body) {
            events.extend(acc.push(&payload));
        }
        (events, acc.finish())
    }

    #[test]
    fn accumulates_a_text_turn_and_ends_on_the_done_sentinel() {
        let body = concat!(
            "data: {\"model\":\"anthropic/claude-sonnet-5\",\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3}}\n\n",
            "data: [DONE]\n\n",
        ).as_bytes();

        let (events, turn) = run(body);
        assert_eq!(turn.text, "Hello world");
        assert_eq!(turn.finish_reason.as_deref(), Some("stop"));
        assert_eq!(turn.model, "anthropic/claude-sonnet-5");
        assert_eq!(turn.usage.completion_tokens, 3);
        assert_eq!(events.last(), Some(&TurnEvent::Done));
    }

    #[test]
    fn the_done_sentinel_is_matched_before_json_parsing() {
        // `[DONE]` is valid JSON (an array), so parsing first would swallow it.
        let mut acc = TurnAccumulator::new();
        assert_eq!(acc.push("[DONE]"), vec![TurnEvent::Done]);
        assert_eq!(acc.push(" [DONE] "), vec![TurnEvent::Done]);
    }

    #[test]
    fn openrouter_keepalive_comments_never_reach_the_accumulator() {
        // OpenRouter emits `: OPENROUTER PROCESSING` while the upstream model
        // works; the shared SSE decoder drops comment frames.
        let body = concat!(
            ": OPENROUTER PROCESSING\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
        )
        .as_bytes();
        let (_, turn) = run(body);
        assert_eq!(turn.text, "ok");
    }

    #[test]
    fn tool_call_arguments_are_rejoined_from_their_fragments() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"run_command\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"commands\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"load 1crn\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        ).as_bytes();

        let (events, turn) = run(body);
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].id, "call_1");
        assert_eq!(turn.tool_calls[0].function.name, "run_command");
        assert_eq!(
            turn.tool_calls[0].function.arguments,
            r#"{"commands":"load 1crn"}"#
        );
        assert!(events.contains(&TurnEvent::ToolUseStarted {
            name: "run_command".to_string()
        }));
    }

    #[test]
    fn parallel_tool_calls_are_kept_apart_by_their_index() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"screenshot\",\"arguments\":\"{}\"}},{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"count_atoms\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].function.name, "screenshot");
        assert_eq!(turn.tool_calls[1].id, "b");
    }

    #[test]
    fn a_tool_use_is_announced_once_even_when_its_name_is_split() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"run_\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"command\"}}]}}]}\n\n",
        ).as_bytes();

        let (events, turn) = run(body);
        assert_eq!(turn.tool_calls[0].function.name, "run_command");
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, TurnEvent::ToolUseStarted { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn a_nameless_tool_slot_from_a_truncated_stream_is_discarded() {
        // Nothing can be executed from it, and passing it on would strand the
        // agent loop waiting for a result that can never be produced.
        let body =
            b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\"}}]}}]}\n\n";
        let (_, turn) = run(body);
        assert!(turn.tool_calls.is_empty());
    }

    #[test]
    fn reasoning_is_signalled_but_never_streamed_into_the_transcript() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"let me think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\" some more\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"the answer\"}}]}\n\n",
        )
        .as_bytes();

        let (events, turn) = run(body);
        assert_eq!(turn.text, "the answer");
        assert_eq!(
            events
                .iter()
                .filter(|e| **e == TurnEvent::ReasoningStarted)
                .count(),
            1
        );
        assert!(!events
            .iter()
            .any(|e| matches!(e, TurnEvent::TextDelta(t) if t.contains("think"))));
    }

    #[test]
    fn reasoning_fragments_rejoin_into_one_block() {
        // Upstream vendors validate the sequence as the model emitted it, so
        // three blocks where the model produced one is a rejected history.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"index\":0,\"text\":\"first \"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"index\":0,\"text\":\"second\"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"index\":0,\"signature\":\"sig-abc\"}]}}]}\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        assert_eq!(turn.reasoning_details.len(), 1);
        assert_eq!(turn.reasoning_details[0]["text"], "first second");
        assert_eq!(turn.reasoning_details[0]["signature"], "sig-abc");
    }

    #[test]
    fn distinct_reasoning_blocks_stay_distinct() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"index\":0,\"text\":\"a\"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.text\",\"index\":1,\"text\":\"b\"}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.encrypted\",\"index\":1,\"data\":\"xx\"}]}}]}\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        assert_eq!(turn.reasoning_details.len(), 3);
    }

    #[test]
    fn surfaces_errors_in_both_shapes_openrouter_uses() {
        let top_level = b"data: {\"error\":{\"code\":429,\"message\":\"Rate limited\"}}\n\n";
        assert_eq!(
            run(top_level).0,
            vec![TurnEvent::Error("Rate limited".to_string())]
        );

        let per_choice =
            b"data: {\"choices\":[{\"error\":{\"message\":\"upstream failed\"},\"delta\":{}}]}\n\n";
        assert!(run(per_choice)
            .0
            .contains(&TurnEvent::Error("upstream failed".to_string())));
    }

    #[test]
    fn unparseable_chunks_are_ignored_rather_than_failing_the_stream() {
        let body = concat!(
            "data: not json\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
        )
        .as_bytes();
        assert_eq!(run(body).1.text, "ok");
    }
}
