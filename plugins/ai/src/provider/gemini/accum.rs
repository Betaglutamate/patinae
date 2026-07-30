//! Assembles Gemini SSE chunks into a completed turn.
//!
//! Gemini streams differently from Anthropic. Each chunk is a *complete*
//! `GenerateContentResponse` carrying whatever parts became available, rather
//! than indexed deltas against blocks opened earlier — so there is no block
//! table here, only concatenation.
//!
//! Two consequences shape this module:
//!
//! - **Text arrives split across chunks and must be rejoined**, but reasoning
//!   text and answer text interleave in the same stream and are distinguished
//!   only by the `thought` flag. Merging across that boundary would paste the
//!   model's private reasoning into its answer, so runs are merged only while
//!   the flag matches.
//! - **`thoughtSignature` may land on a later chunk than the text it signs.**
//!   The merged run keeps the last signature seen, because it signs the run.

use super::types::{AssistantTurn, FinishReason, GenerateContentResponse, Part, UsageMetadata};

/// Something worth surfacing to the UI as the turn streams in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnEvent {
    TextDelta(String),
    ThinkingStarted,
    ToolUseStarted { name: String },
    Error(String),
    Done,
}

#[derive(Debug, Default)]
pub struct TurnAccumulator {
    parts: Vec<Part>,
    finish_reason: Option<FinishReason>,
    block_reason: Option<String>,
    usage: UsageMetadata,
    model: String,
    thinking_announced: bool,
}

impl TurnAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one SSE `data` payload.
    ///
    /// A chunk that fails to parse is skipped rather than failing the stream: a
    /// field added server-side must not be able to break an in-flight agent
    /// loop.
    pub fn push(&mut self, data: &str) -> Vec<TurnEvent> {
        // Errors arrive as a plain object with no candidates, so check before
        // committing to the response shape.
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(message) = v
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
            {
                return vec![TurnEvent::Error(message.to_string())];
            }
        }

        let Ok(chunk) = serde_json::from_str::<GenerateContentResponse>(data) else {
            return vec![];
        };

        if let Some(m) = chunk.model_version {
            self.model = m;
        }
        if let Some(u) = chunk.usage_metadata {
            self.usage = u;
        }
        if let Some(reason) = chunk.prompt_feedback.and_then(|f| f.block_reason) {
            self.block_reason = Some(reason);
        }

        let mut events = Vec::new();
        for candidate in chunk.candidates {
            if let Some(reason) = candidate.finish_reason {
                self.finish_reason = Some(reason);
            }
            let Some(content) = candidate.content else {
                continue;
            };
            for part in content.parts {
                events.extend(self.absorb(part));
            }
        }
        events
    }

    /// Fold one incoming part into the accumulated turn.
    fn absorb(&mut self, part: Part) -> Vec<TurnEvent> {
        match part {
            Part::Text {
                text,
                thought,
                thought_signature,
            } => {
                let is_thought = thought == Some(true);
                let mut events = Vec::new();
                if is_thought {
                    if !self.thinking_announced {
                        self.thinking_announced = true;
                        events.push(TurnEvent::ThinkingStarted);
                    }
                } else if !text.is_empty() {
                    events.push(TurnEvent::TextDelta(text.clone()));
                }
                self.append_text(text, is_thought, thought_signature);
                events
            }
            Part::FunctionCall {
                function_call,
                thought_signature,
            } => {
                let name = function_call.name.clone();
                self.parts.push(Part::FunctionCall {
                    function_call,
                    thought_signature,
                });
                vec![TurnEvent::ToolUseStarted { name }]
            }
            // The model does not emit these, but keep the turn lossless rather
            // than silently dropping something we did not expect.
            other => {
                self.parts.push(other);
                vec![]
            }
        }
    }

    /// Extend the trailing text run, or start a new one when the thought flag
    /// changes.
    fn append_text(&mut self, text: String, is_thought: bool, signature: Option<String>) {
        if let Some(Part::Text {
            text: buf,
            thought,
            thought_signature,
        }) = self.parts.last_mut()
        {
            if (*thought == Some(true)) == is_thought {
                buf.push_str(&text);
                // A signature arriving late signs the whole run; an absent one
                // must not erase a signature already recorded for it.
                if signature.is_some() {
                    *thought_signature = signature;
                }
                return;
            }
        }
        self.parts.push(Part::Text {
            text,
            thought: is_thought.then_some(true),
            thought_signature: signature,
        });
    }

    pub fn finish(self) -> AssistantTurn {
        AssistantTurn {
            // A trailing empty text part is what a stream that ended on a tool
            // call leaves behind; the API rejects it on the next request.
            parts: self
                .parts
                .into_iter()
                .filter(|p| !matches!(p, Part::Text { text, thought_signature: None, .. } if text.is_empty()))
                .collect(),
            finish_reason: self.finish_reason,
            block_reason: self.block_reason,
            usage: self.usage,
            model: self.model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::SseDecoder;

    /// Drive the accumulator over a raw SSE body, exactly as the worker will.
    fn run(body: &[u8]) -> (Vec<TurnEvent>, AssistantTurn) {
        let mut decoder = SseDecoder::new();
        let mut acc = TurnAccumulator::new();
        let mut events = Vec::new();
        for payload in decoder.push(body) {
            events.extend(acc.push(&payload));
        }
        (events, acc.finish())
    }

    fn text_of(turn: &AssistantTurn) -> String {
        turn.parts
            .iter()
            .filter_map(|p| match p {
                Part::Text {
                    text,
                    thought: None,
                    ..
                } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn rejoins_text_split_across_chunks() {
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello \"}]}}],\"modelVersion\":\"gemini-2.5-pro\"}\n\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"world\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2}}\n\n",
        ).as_bytes();

        let (events, turn) = run(body);
        assert_eq!(text_of(&turn), "Hello world");
        assert_eq!(
            turn.parts.len(),
            1,
            "one merged run, not one part per chunk"
        );
        assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
        assert_eq!(turn.model, "gemini-2.5-pro");
        assert_eq!(turn.usage.prompt_tokens, 5);
        assert!(events.contains(&TurnEvent::TextDelta("Hello ".to_string())));
    }

    #[test]
    fn reasoning_text_never_merges_into_the_answer() {
        // Both are `text` parts in one stream; only the flag separates them.
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"let me think\",\"thought\":true}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"the answer\"}]}}]}\n\n",
        ).as_bytes();

        let (events, turn) = run(body);
        assert_eq!(text_of(&turn), "the answer");
        assert_eq!(turn.parts.len(), 2);
        assert!(turn.parts[0].is_thought());
        assert!(events.contains(&TurnEvent::ThinkingStarted));
        // Reasoning must not be streamed to the transcript as prose.
        assert!(!events.contains(&TurnEvent::TextDelta("let me think".to_string())));
    }

    #[test]
    fn thinking_is_announced_once_however_many_thought_chunks_arrive() {
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\",\"thought\":true}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"b\",\"thought\":true}]}}]}\n\n",
        ).as_bytes();

        let (events, _) = run(body);
        assert_eq!(
            events
                .iter()
                .filter(|e| **e == TurnEvent::ThinkingStarted)
                .count(),
            1
        );
    }

    #[test]
    fn a_signature_arriving_after_its_text_is_still_attached() {
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"think\",\"thought\":true}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"\",\"thought\":true,\"thoughtSignature\":\"sig-xyz\"}]}}]}\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        match &turn.parts[0] {
            Part::Text {
                text,
                thought_signature,
                ..
            } => {
                assert_eq!(text, "think");
                assert_eq!(thought_signature.as_deref(), Some("sig-xyz"));
            }
            other => panic!("expected a thought run, got {other:?}"),
        }
    }

    #[test]
    fn a_later_chunk_without_a_signature_does_not_erase_one_already_seen() {
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\",\"thought\":true,\"thoughtSignature\":\"sig\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"b\",\"thought\":true}]}}]}\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        match &turn.parts[0] {
            Part::Text {
                thought_signature, ..
            } => assert_eq!(thought_signature.as_deref(), Some("sig")),
            other => panic!("expected a thought run, got {other:?}"),
        }
    }

    #[test]
    fn function_calls_are_captured_whole_with_their_signature() {
        let body = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"run_command\",\"args\":{\"commands\":\"load 1crn\"}},\"thoughtSignature\":\"sig\"}]},\"finishReason\":\"STOP\"}]}\n\n".as_bytes();

        let (events, turn) = run(body);
        assert!(events.contains(&TurnEvent::ToolUseStarted {
            name: "run_command".to_string()
        }));
        match &turn.parts[0] {
            Part::FunctionCall {
                function_call,
                thought_signature,
            } => {
                assert_eq!(function_call.args["commands"], "load 1crn");
                assert_eq!(thought_signature.as_deref(), Some("sig"));
            }
            other => panic!("expected a function call, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_trailing_text_part_is_dropped() {
        // The API rejects an empty text part when the turn is replayed.
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"screenshot\",\"args\":{}}}]}}]}\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        assert_eq!(turn.parts.len(), 1);
        assert!(matches!(turn.parts[0], Part::FunctionCall { .. }));
    }

    #[test]
    fn a_blocked_prompt_is_recorded_even_with_no_candidates() {
        let body =
            b"data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\"},\"usageMetadata\":{\"promptTokenCount\":3}}\n\n";
        let (_, turn) = run(body);
        assert_eq!(turn.block_reason.as_deref(), Some("SAFETY"));
        assert!(turn.parts.is_empty());
    }

    #[test]
    fn surfaces_mid_stream_errors() {
        let body =
            b"data: {\"error\":{\"code\":429,\"message\":\"Resource exhausted\",\"status\":\"RESOURCE_EXHAUSTED\"}}\n\n";
        let (events, _) = run(body);
        assert_eq!(
            events,
            vec![TurnEvent::Error("Resource exhausted".to_string())]
        );
    }

    #[test]
    fn unparseable_and_unknown_chunks_are_ignored() {
        let body = concat!(
            "data: not json at all\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]}}]}\n\n",
        )
        .as_bytes();
        let (_, turn) = run(body);
        assert_eq!(text_of(&turn), "ok");
    }
}
