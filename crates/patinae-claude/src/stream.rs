//! Server-sent event parsing and message accumulation.
//!
//! Split out from the HTTP transport so the fiddly part — reassembling frames
//! that arrive split across arbitrary byte boundaries, and rebuilding a
//! complete assistant turn from deltas — is pure and testable without a
//! network.
//!
//! Two stages:
//!
//! 1. [`SseParser`] turns a byte stream into [`StreamEvent`]s.
//! 2. [`MessageAccumulator`] folds those events into an [`AssistantTurn`],
//!    emitting [`StreamUpdate`]s along the way so the UI can render tokens as
//!    they arrive.

use serde::Deserialize;

use crate::types::{ContentBlock, StopDetails, StopReason, Usage};

/// An event from the Messages API stream.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Start of a message, carrying initial usage.
    MessageStart {
        /// The message envelope.
        message: MessageStart,
    },
    /// A new content block begins.
    ContentBlockStart {
        /// Position within the message content.
        index: usize,
        /// The block, with empty/partial payload.
        content_block: ContentBlock,
    },
    /// Incremental content for an open block.
    ContentBlockDelta {
        /// Position within the message content.
        index: usize,
        /// The increment.
        delta: BlockDelta,
    },
    /// An open content block is complete.
    ContentBlockStop {
        /// Position within the message content.
        index: usize,
    },
    /// Message-level update carrying the stop reason and final output usage.
    MessageDelta {
        /// The message-level increment.
        delta: MessageDelta,
        /// Output token count for the turn.
        #[serde(default)]
        usage: Option<Usage>,
    },
    /// The message is complete.
    MessageStop,
    /// Keepalive.
    Ping,
    /// A server-reported error.
    Error {
        /// Error detail.
        error: ApiError,
    },
    /// An event type this build does not model.
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

/// Message envelope from `message_start`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MessageStart {
    /// Model that produced the message.
    #[serde(default)]
    pub model: Option<String>,
    /// Input-side usage, known up front.
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Message-level increment from `message_delta`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct MessageDelta {
    /// Why generation stopped.
    #[serde(default)]
    pub stop_reason: Option<StopReason>,
    /// Structured refusal detail, populated only on a refusal.
    #[serde(default)]
    pub stop_details: Option<StopDetails>,
}

/// An increment to an open content block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockDelta {
    /// Appended answer text.
    TextDelta {
        /// The text fragment.
        text: String,
    },
    /// Appended fragment of a tool input's JSON.
    InputJsonDelta {
        /// Raw JSON fragment; only valid once concatenated.
        partial_json: String,
    },
    /// Appended reasoning text.
    ThinkingDelta {
        /// The reasoning fragment.
        thinking: String,
    },
    /// The signature for a thinking block.
    SignatureDelta {
        /// Opaque signature.
        signature: String,
    },
    /// A delta type this build does not model.
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

/// A server-reported error.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ApiError {
    /// Error type, e.g. `overloaded_error`.
    #[serde(rename = "type", default)]
    pub kind: String,
    /// Human-readable message.
    #[serde(default)]
    pub message: String,
}

/// Incremental parser for a Server-Sent Events byte stream.
///
/// Chunks arrive on arbitrary boundaries, so a frame may be split mid-JSON or
/// even mid-line. Feed every chunk to [`SseParser::push`] and it will emit
/// events only once complete frames are available.
#[derive(Debug, Default)]
pub struct SseParser {
    buffer: String,
}

impl SseParser {
    /// Creates an empty parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds a chunk and returns any events it completed.
    ///
    /// Malformed frames are skipped rather than surfaced as errors: a single
    /// unparseable frame should not abort a response that is otherwise fine.
    pub fn push(&mut self, chunk: &str) -> Vec<StreamEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        // Frames are separated by a blank line. Normalising CRLF first keeps
        // the split logic to a single case.
        if self.buffer.contains('\r') {
            self.buffer = self.buffer.replace("\r\n", "\n");
        }

        while let Some(end) = self.buffer.find("\n\n") {
            let frame: String = self.buffer.drain(..end + 2).collect();
            if let Some(event) = parse_frame(&frame) {
                events.push(event);
            }
        }
        events
    }

    /// Returns `true` when no partial frame is buffered.
    pub fn is_empty(&self) -> bool {
        self.buffer.trim().is_empty()
    }
}

/// Extracts the `data:` payload from one SSE frame and parses it.
fn parse_frame(frame: &str) -> Option<StreamEvent> {
    let mut data = String::new();
    for line in frame.lines() {
        // The `event:` line duplicates the JSON's own `type` field, so only
        // `data:` matters. Comments (`:` prefix) are ignored.
        let Some(rest) = line.strip_prefix("data:") else {
            continue;
        };
        // SSE strips exactly one leading space after the colon.
        let rest = rest.strip_prefix(' ').unwrap_or(rest);
        if !data.is_empty() {
            data.push('\n');
        }
        data.push_str(rest);
    }

    if data.is_empty() {
        return None;
    }
    serde_json::from_str(&data).ok()
}

/// A user-visible change produced by folding a [`StreamEvent`].
#[derive(Debug, Clone, PartialEq)]
pub enum StreamUpdate {
    /// Append this fragment to the visible answer.
    Text(String),
    /// Append this fragment to the reasoning display.
    Thinking(String),
    /// The model began requesting a tool.
    ToolUseStarted {
        /// Correlation id.
        id: String,
        /// Tool name.
        name: String,
    },
    /// The turn finished.
    Completed,
    /// The server reported an error mid-stream.
    Failed(ApiError),
}

/// A complete assistant turn rebuilt from a stream.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AssistantTurn {
    /// Content blocks, in order, ready to echo back as conversation history.
    pub content: Vec<ContentBlock>,
    /// Why generation stopped.
    pub stop_reason: Option<StopReason>,
    /// Refusal detail, when the turn was declined.
    pub stop_details: Option<StopDetails>,
    /// Token accounting.
    pub usage: Usage,
    /// Model that actually served the turn (may differ under fallback).
    pub model: Option<String>,
}

impl AssistantTurn {
    /// Returns the tool calls the model is waiting on.
    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.as_str(), name.as_str(), input))
                }
                _ => None,
            })
            .collect()
    }

    /// Concatenated visible text.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(ContentBlock::as_text)
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns `true` when the model is waiting for tool results.
    pub fn wants_tools(&self) -> bool {
        self.stop_reason == Some(StopReason::ToolUse)
    }
}

/// A content block still being streamed.
#[derive(Debug, Clone)]
enum PartialBlock {
    Text(String),
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
    /// A block that needs no accumulation, kept verbatim.
    Complete(ContentBlock),
}

/// Folds [`StreamEvent`]s into an [`AssistantTurn`].
#[derive(Debug, Default)]
pub struct MessageAccumulator {
    blocks: Vec<PartialBlock>,
    stop_reason: Option<StopReason>,
    stop_details: Option<StopDetails>,
    usage: Usage,
    model: Option<String>,
}

impl MessageAccumulator {
    /// Creates an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds one event, returning a user-visible update when there is one.
    pub fn push(&mut self, event: StreamEvent) -> Option<StreamUpdate> {
        match event {
            StreamEvent::MessageStart { message } => {
                self.model = message.model;
                if let Some(usage) = message.usage {
                    // message_start carries input-side counts; output arrives
                    // later on message_delta.
                    self.usage.input_tokens = usage.input_tokens;
                    self.usage.cache_creation_input_tokens = usage.cache_creation_input_tokens;
                    self.usage.cache_read_input_tokens = usage.cache_read_input_tokens;
                }
                None
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let (partial, update) = match content_block {
                    ContentBlock::Text { text, .. } => {
                        let update = (!text.is_empty()).then(|| StreamUpdate::Text(text.clone()));
                        (PartialBlock::Text(text), update)
                    }
                    ContentBlock::Thinking {
                        thinking,
                        signature,
                    } => {
                        let update = (!thinking.is_empty())
                            .then(|| StreamUpdate::Thinking(thinking.clone()));
                        (
                            PartialBlock::Thinking {
                                thinking,
                                signature,
                            },
                            update,
                        )
                    }
                    ContentBlock::ToolUse { id, name, .. } => {
                        let update = StreamUpdate::ToolUseStarted {
                            id: id.clone(),
                            name: name.clone(),
                        };
                        (
                            PartialBlock::ToolUse {
                                id,
                                name,
                                json: String::new(),
                            },
                            Some(update),
                        )
                    }
                    other => (PartialBlock::Complete(other), None),
                };
                self.set_block(index, partial);
                update
            }
            StreamEvent::ContentBlockDelta { index, delta } => self.apply_delta(index, delta),
            StreamEvent::ContentBlockStop { .. } => None,
            StreamEvent::MessageDelta { delta, usage } => {
                if let Some(reason) = delta.stop_reason {
                    self.stop_reason = Some(reason);
                }
                if let Some(details) = delta.stop_details {
                    self.stop_details = Some(details);
                }
                if let Some(usage) = usage {
                    self.usage.output_tokens = usage.output_tokens;
                }
                None
            }
            StreamEvent::MessageStop => Some(StreamUpdate::Completed),
            StreamEvent::Ping | StreamEvent::Unknown(_) => None,
            StreamEvent::Error { error } => Some(StreamUpdate::Failed(error)),
        }
    }

    /// Consumes the accumulator, producing the finished turn.
    ///
    /// Tool inputs are parsed here rather than incrementally, because a
    /// partial JSON fragment is not valid JSON on its own. A tool call with no
    /// arguments streams no deltas at all, so an empty buffer becomes `{}`
    /// rather than an error.
    pub fn finish(self) -> AssistantTurn {
        let content = self
            .blocks
            .into_iter()
            .map(|block| match block {
                PartialBlock::Text(text) => ContentBlock::Text {
                    text,
                    cache_control: None,
                },
                PartialBlock::Thinking {
                    thinking,
                    signature,
                } => ContentBlock::Thinking {
                    thinking,
                    signature,
                },
                PartialBlock::ToolUse { id, name, json } => {
                    let input = if json.trim().is_empty() {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str(&json).unwrap_or_else(|_| serde_json::json!({}))
                    };
                    ContentBlock::ToolUse { id, name, input }
                }
                PartialBlock::Complete(block) => block,
            })
            .collect();

        AssistantTurn {
            content,
            stop_reason: self.stop_reason,
            stop_details: self.stop_details,
            usage: self.usage,
            model: self.model,
        }
    }

    fn set_block(&mut self, index: usize, block: PartialBlock) {
        if self.blocks.len() <= index {
            // Blocks normally arrive in order, but do not assume it.
            self.blocks
                .resize_with(index + 1, || PartialBlock::Text(String::new()));
        }
        self.blocks[index] = block;
    }

    fn apply_delta(&mut self, index: usize, delta: BlockDelta) -> Option<StreamUpdate> {
        let block = self.blocks.get_mut(index)?;
        match (block, delta) {
            (PartialBlock::Text(buf), BlockDelta::TextDelta { text }) => {
                buf.push_str(&text);
                Some(StreamUpdate::Text(text))
            }
            (
                PartialBlock::Thinking { thinking, .. },
                BlockDelta::ThinkingDelta { thinking: t },
            ) => {
                thinking.push_str(&t);
                Some(StreamUpdate::Thinking(t))
            }
            (
                PartialBlock::Thinking { signature, .. },
                BlockDelta::SignatureDelta { signature: s },
            ) => {
                *signature = Some(s);
                None
            }
            (PartialBlock::ToolUse { json, .. }, BlockDelta::InputJsonDelta { partial_json }) => {
                json.push_str(&partial_json);
                None
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives a parser with chunks and returns every event produced.
    fn parse_all(chunks: &[&str]) -> Vec<StreamEvent> {
        let mut parser = SseParser::new();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.push(chunk));
        }
        events
    }

    #[test]
    fn parses_a_simple_frame() {
        let events = parse_all(&["event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"]);
        assert_eq!(events, vec![StreamEvent::MessageStop]);
    }

    #[test]
    fn parses_multiple_frames_in_one_chunk() {
        let events = parse_all(&[concat!(
            "data: {\"type\":\"ping\"}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n"
        )]);
        assert_eq!(events, vec![StreamEvent::Ping, StreamEvent::MessageStop]);
    }

    #[test]
    fn reassembles_a_frame_split_mid_json() {
        // The realistic failure mode: a chunk boundary lands inside the payload.
        let events = parse_all(&[
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"tex",
            "t_delta\",\"text\":\"hello\"}}\n",
            "\n",
        ]);

        assert_eq!(
            events,
            vec![StreamEvent::ContentBlockDelta {
                index: 0,
                delta: BlockDelta::TextDelta {
                    text: "hello".to_string()
                },
            }]
        );
    }

    #[test]
    fn emits_nothing_until_a_frame_is_complete() {
        let mut parser = SseParser::new();

        assert!(parser.push("data: {\"type\":\"ping\"}").is_empty());
        assert!(!parser.is_empty(), "partial frame should stay buffered");
        assert_eq!(parser.push("\n\n"), vec![StreamEvent::Ping]);
        assert!(parser.is_empty());
    }

    #[test]
    fn handles_crlf_line_endings() {
        let events = parse_all(&["data: {\"type\":\"ping\"}\r\n\r\n"]);
        assert_eq!(events, vec![StreamEvent::Ping]);
    }

    #[test]
    fn tolerates_missing_space_after_the_colon() {
        let events = parse_all(&["data:{\"type\":\"ping\"}\n\n"]);
        assert_eq!(events, vec![StreamEvent::Ping]);
    }

    #[test]
    fn skips_malformed_frames_without_aborting_the_stream() {
        let events = parse_all(&[concat!(
            "data: {not json\n\n",
            "data: {\"type\":\"message_stop\"}\n\n"
        )]);

        // The bad frame is dropped; the good one still arrives.
        assert_eq!(events, vec![StreamEvent::MessageStop]);
    }

    #[test]
    fn ignores_comment_lines() {
        let events = parse_all(&[": keepalive comment\ndata: {\"type\":\"ping\"}\n\n"]);
        assert_eq!(events, vec![StreamEvent::Ping]);
    }

    #[test]
    fn unknown_event_types_do_not_abort_the_stream() {
        let events = parse_all(&[concat!(
            "data: {\"type\":\"future_event\",\"x\":1}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n"
        )]);

        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], StreamEvent::Unknown(_)));
        assert_eq!(events[1], StreamEvent::MessageStop);
    }

    /// A complete, realistic text-only turn.
    fn text_turn() -> Vec<&'static str> {
        vec![
            r#"data: {"type":"message_start","message":{"model":"claude-opus-5","usage":{"input_tokens":10,"cache_read_input_tokens":900}}}"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Show"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ing cartoon."}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]
    }

    fn accumulate(lines: &[&str]) -> (Vec<StreamUpdate>, AssistantTurn) {
        let mut parser = SseParser::new();
        let mut acc = MessageAccumulator::new();
        let mut updates = Vec::new();
        for line in lines {
            for event in parser.push(&format!("{line}\n\n")) {
                if let Some(update) = acc.push(event) {
                    updates.push(update);
                }
            }
        }
        (updates, acc.finish())
    }

    #[test]
    fn accumulates_a_text_turn() {
        let (updates, turn) = accumulate(&text_turn());

        assert_eq!(turn.text(), "Showing cartoon.");
        assert_eq!(turn.stop_reason, Some(StopReason::EndTurn));
        assert!(!turn.wants_tools());
        assert_eq!(turn.model.as_deref(), Some("claude-opus-5"));
        assert_eq!(
            updates,
            vec![
                StreamUpdate::Text("Show".to_string()),
                StreamUpdate::Text("ing cartoon.".to_string()),
                StreamUpdate::Completed,
            ]
        );
    }

    #[test]
    fn tracks_usage_across_start_and_delta() {
        let (_, turn) = accumulate(&text_turn());

        assert_eq!(turn.usage.input_tokens, 10);
        assert_eq!(turn.usage.cache_read_input_tokens, 900);
        assert_eq!(turn.usage.output_tokens, 7);
        assert_eq!(turn.usage.total_input_tokens(), 910);
    }

    #[test]
    fn accumulates_tool_input_from_json_fragments() {
        let (updates, turn) = accumulate(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"run_command","input":{}}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"comm"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"and\": \"show cartoon\"}"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);

        assert!(turn.wants_tools());
        let tools = turn.tool_uses();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].0, "toolu_1");
        assert_eq!(tools[0].1, "run_command");
        assert_eq!(tools[0].2["command"], "show cartoon");

        assert!(updates.contains(&StreamUpdate::ToolUseStarted {
            id: "toolu_1".to_string(),
            name: "run_command".to_string(),
        }));
    }

    #[test]
    fn a_tool_call_with_no_arguments_yields_an_empty_object() {
        // No input_json_delta events at all; the buffer stays empty.
        let (_, turn) = accumulate(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"list_objects","input":{}}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        ]);

        let tools = turn.tool_uses();
        assert_eq!(tools[0].2, &serde_json::json!({}));
    }

    #[test]
    fn malformed_tool_json_degrades_to_an_empty_object() {
        let (_, turn) = accumulate(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"x","input":{}}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"broken"}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);

        // Better a tool call with no args (which the tool can reject clearly)
        // than a panic or a dropped turn.
        assert_eq!(turn.tool_uses()[0].2, &serde_json::json!({}));
    }

    #[test]
    fn interleaves_thinking_and_text_blocks() {
        let (updates, turn) = accumulate(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Consider cartoon."}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-abc"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Done."}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);

        assert_eq!(turn.text(), "Done.");
        assert_eq!(turn.content.len(), 2);
        // The signature must survive so the block can be echoed back unchanged.
        assert_eq!(
            turn.content[0],
            ContentBlock::Thinking {
                thinking: "Consider cartoon.".to_string(),
                signature: Some("sig-abc".to_string()),
            }
        );
        assert!(updates.contains(&StreamUpdate::Thinking("Consider cartoon.".to_string())));
    }

    #[test]
    fn captures_refusal_stop_reason_and_details() {
        let (_, turn) = accumulate(&[
            r#"data: {"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"category":"bio","explanation":"declined"}}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);

        assert_eq!(turn.stop_reason, Some(StopReason::Refusal));
        let details = turn.stop_details.expect("refusal should carry details");
        assert_eq!(details.category.as_deref(), Some("bio"));
        assert_eq!(details.explanation.as_deref(), Some("declined"));
    }

    #[test]
    fn max_tokens_truncation_is_visible_on_the_turn() {
        let (_, turn) = accumulate(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}"#,
        ]);

        assert_eq!(turn.stop_reason, Some(StopReason::MaxTokens));
        assert_eq!(turn.text(), "partial");
    }

    #[test]
    fn server_errors_surface_as_a_failed_update() {
        let (updates, _) = accumulate(&[
            r#"data: {"type":"error","error":{"type":"overloaded_error","message":"try later"}}"#,
        ]);

        assert_eq!(
            updates,
            vec![StreamUpdate::Failed(ApiError {
                kind: "overloaded_error".to_string(),
                message: "try later".to_string(),
            })]
        );
    }

    #[test]
    fn a_delta_for_an_unopened_block_is_ignored() {
        // Defensive: out-of-order or dropped content_block_start must not panic.
        let mut acc = MessageAccumulator::new();
        let update = acc.push(StreamEvent::ContentBlockDelta {
            index: 7,
            delta: BlockDelta::TextDelta {
                text: "orphan".to_string(),
            },
        });

        assert!(update.is_none());
        assert!(acc.finish().content.is_empty());
    }
}
