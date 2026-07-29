//! Incremental Server-Sent Events decoder.
//!
//! There is no official Anthropic Rust SDK, so the `POST /v1/messages` SSE
//! stream is decoded here by hand. The decoder is byte-oriented on purpose:
//! HTTP chunk boundaries can split a multi-byte UTF-8 character, so we buffer
//! raw bytes and only convert to `str` once a complete frame has been framed
//! off (frame delimiters are ASCII, so splitting on bytes is always safe).

/// Buffers partial SSE frames across chunk boundaries and yields the `data`
/// payload of each complete event.
///
/// Per the SSE spec: events are separated by a blank line, `:`-prefixed lines
/// are comments (the API uses them as heartbeats), and multiple `data:` lines
/// within one event are joined with newlines.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buf: Vec<u8>,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next chunk of the response body, returning the `data` payload
    /// of every event that became complete.
    ///
    /// A chunk that ends mid-frame contributes nothing and is retained until
    /// the rest arrives.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();

        while let Some((frame_end, next_start)) = find_frame_boundary(&self.buf) {
            let frame = self.buf[..frame_end].to_vec();
            self.buf.drain(..next_start);
            if let Some(data) = parse_frame(&frame) {
                out.push(data);
            }
        }
        out
    }
}

/// Locate the first frame terminator, returning `(end_of_frame, start_of_next)`.
///
/// Handles both `\n\n` and `\r\n\r\n` so the decoder does not depend on the
/// server's line-ending choice.
fn find_frame_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let mut lf = None;
    let mut crlf = None;
    for i in 0..buf.len() {
        if lf.is_none() && buf[i..].starts_with(b"\n\n") {
            lf = Some((i, i + 2));
        }
        if crlf.is_none() && buf[i..].starts_with(b"\r\n\r\n") {
            crlf = Some((i, i + 4));
        }
        if lf.is_some() && crlf.is_some() {
            break;
        }
    }
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Extract the joined `data:` payload from one frame, or `None` for
/// comment-only / dataless frames (heartbeats).
fn parse_frame(frame: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(frame).ok()?;
    let mut data: Vec<&str> = Vec::new();

    for line in text.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        // `event:` names the type, but every Anthropic payload also carries a
        // `type` field, so the data line alone is sufficient.
        if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value));
        }
    }

    if data.is_empty() {
        None
    } else {
        Some(data.join("\n"))
    }
}

// =============================================================================
// Turn accumulation
// =============================================================================

use crate::api::types::{AssistantTurn, ContentBlock, StopDetails, StopReason, Usage};

/// Something worth surfacing to the UI as the turn streams in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnEvent {
    /// Assistant prose arrived. Append to the transcript.
    TextDelta(String),
    /// The model began thinking. On `claude-opus-5` the content is omitted by
    /// default, so this is a progress signal, not text.
    ThinkingStarted,
    /// The model began requesting a tool call.
    ToolUseStarted { name: String },
    /// The API reported an error mid-stream.
    Error(String),
    /// `message_stop` — the turn is complete.
    Done,
}

/// A content block still being streamed.
#[derive(Debug, Clone)]
enum Partial {
    Text(String),
    Thinking { thinking: String, signature: String },
    RedactedThinking { data: String },
    ToolUse { id: String, name: String, json: String },
}

/// Assembles SSE events into a complete [`AssistantTurn`].
///
/// Parsing is deliberately `Value`-based rather than strictly typed: unknown
/// event types and unknown delta types are ignored rather than failing the
/// stream, so a server-side addition cannot break an in-flight agent loop.
#[derive(Debug, Default)]
pub struct TurnAccumulator {
    blocks: Vec<Option<Partial>>,
    stop_reason: Option<StopReason>,
    stop_details: Option<StopDetails>,
    usage: Usage,
    model: String,
}

impl TurnAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one SSE `data` payload.
    pub fn push(&mut self, data: &str) -> Vec<TurnEvent> {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
            return vec![];
        };
        let Some(kind) = v.get("type").and_then(|t| t.as_str()) else {
            return vec![];
        };

        match kind {
            "message_start" => {
                if let Some(msg) = v.get("message") {
                    if let Some(m) = msg.get("model").and_then(|m| m.as_str()) {
                        self.model = m.to_string();
                    }
                    if let Some(u) = msg.get("usage") {
                        if let Ok(u) = serde_json::from_value::<Usage>(u.clone()) {
                            self.usage = u;
                        }
                    }
                }
                vec![]
            }
            "content_block_start" => self.on_block_start(&v),
            "content_block_delta" => self.on_block_delta(&v),
            "message_delta" => {
                if let Some(d) = v.get("delta") {
                    if let Some(sr) = d.get("stop_reason") {
                        self.stop_reason = serde_json::from_value(sr.clone()).ok();
                    }
                    if let Some(sd) = d.get("stop_details") {
                        self.stop_details = serde_json::from_value(sd.clone()).ok();
                    }
                }
                // `message_delta.usage` carries the final output token count and
                // must not clobber the input counts from `message_start`.
                if let Some(u) = v.get("usage") {
                    if let Some(n) = u.get("output_tokens").and_then(|n| n.as_u64()) {
                        self.usage.output_tokens = n as u32;
                    }
                }
                vec![]
            }
            "message_stop" => vec![TurnEvent::Done],
            "error" => {
                let msg = v
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown streaming error");
                vec![TurnEvent::Error(msg.to_string())]
            }
            _ => vec![],
        }
    }

    fn slot(&mut self, index: usize) -> &mut Option<Partial> {
        if self.blocks.len() <= index {
            self.blocks.resize_with(index + 1, || None);
        }
        &mut self.blocks[index]
    }

    fn on_block_start(&mut self, v: &serde_json::Value) -> Vec<TurnEvent> {
        let index = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
        let Some(cb) = v.get("content_block") else {
            return vec![];
        };
        let kind = cb.get("type").and_then(|t| t.as_str()).unwrap_or_default();

        let (partial, event) = match kind {
            "text" => {
                let text = cb
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or_default()
                    .to_string();
                (Partial::Text(text), None)
            }
            "thinking" => (
                Partial::Thinking {
                    thinking: cb
                        .get("thinking")
                        .and_then(|t| t.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    signature: cb
                        .get("signature")
                        .and_then(|t| t.as_str())
                        .unwrap_or_default()
                        .to_string(),
                },
                Some(TurnEvent::ThinkingStarted),
            ),
            "redacted_thinking" => (
                Partial::RedactedThinking {
                    data: cb
                        .get("data")
                        .and_then(|t| t.as_str())
                        .unwrap_or_default()
                        .to_string(),
                },
                None,
            ),
            "tool_use" => {
                let name = cb
                    .get("name")
                    .and_then(|t| t.as_str())
                    .unwrap_or_default()
                    .to_string();
                (
                    Partial::ToolUse {
                        id: cb
                            .get("id")
                            .and_then(|t| t.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        name: name.clone(),
                        json: String::new(),
                    },
                    Some(TurnEvent::ToolUseStarted { name }),
                )
            }
            _ => return vec![],
        };

        *self.slot(index) = Some(partial);
        event.into_iter().collect()
    }

    fn on_block_delta(&mut self, v: &serde_json::Value) -> Vec<TurnEvent> {
        let index = v.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
        let Some(delta) = v.get("delta") else {
            return vec![];
        };
        let kind = delta
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or_default();
        let text_of = |key: &str| {
            delta
                .get(key)
                .and_then(|t| t.as_str())
                .unwrap_or_default()
                .to_string()
        };

        match kind {
            "text_delta" => {
                let chunk = text_of("text");
                if let Some(Partial::Text(buf)) = self.slot(index) {
                    buf.push_str(&chunk);
                }
                vec![TurnEvent::TextDelta(chunk)]
            }
            "thinking_delta" => {
                let chunk = text_of("thinking");
                if let Some(Partial::Thinking { thinking, .. }) = self.slot(index) {
                    thinking.push_str(&chunk);
                }
                vec![]
            }
            "signature_delta" => {
                let chunk = text_of("signature");
                if let Some(Partial::Thinking { signature, .. }) = self.slot(index) {
                    signature.push_str(&chunk);
                }
                vec![]
            }
            "input_json_delta" => {
                let chunk = text_of("partial_json");
                if let Some(Partial::ToolUse { json, .. }) = self.slot(index) {
                    json.push_str(&chunk);
                }
                vec![]
            }
            _ => vec![],
        }
    }

    /// Finalize into an assistant turn suitable for appending to the
    /// conversation. Tool inputs that never arrived become `{}` rather than
    /// failing the turn.
    pub fn finish(self) -> AssistantTurn {
        let content = self
            .blocks
            .into_iter()
            .flatten()
            .map(|p| match p {
                Partial::Text(text) => ContentBlock::Text { text },
                Partial::Thinking {
                    thinking,
                    signature,
                } => ContentBlock::Thinking {
                    thinking,
                    signature,
                },
                Partial::RedactedThinking { data } => ContentBlock::RedactedThinking { data },
                Partial::ToolUse { id, name, json } => ContentBlock::ToolUse {
                    id,
                    name,
                    input: serde_json::from_str(json.trim())
                        .unwrap_or_else(|_| serde_json::json!({})),
                },
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_single_event() {
        let mut d = SseDecoder::new();
        let out = d.push(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
        assert_eq!(out, vec![r#"{"type":"message_stop"}"#]);
    }

    #[test]
    fn decodes_multiple_events_in_one_chunk() {
        let mut d = SseDecoder::new();
        let out = d.push(b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\n");
        assert_eq!(out, vec![r#"{"a":1}"#, r#"{"b":2}"#]);
    }

    #[test]
    fn buffers_a_frame_split_across_chunks() {
        let mut d = SseDecoder::new();
        assert!(d.push(b"data: {\"type\":\"content_bl").is_empty());
        assert!(d.push(b"ock_delta\"}").is_empty());
        let out = d.push(b"\n\n");
        assert_eq!(out, vec![r#"{"type":"content_block_delta"}"#]);
    }

    #[test]
    fn buffers_a_terminator_split_across_chunks() {
        let mut d = SseDecoder::new();
        assert!(d.push(b"data: {\"x\":1}\n").is_empty());
        let out = d.push(b"\ndata: {\"y\":2}\n\n");
        assert_eq!(out, vec![r#"{"x":1}"#, r#"{"y":2}"#]);
    }

    #[test]
    fn survives_utf8_split_across_chunks() {
        // "α" is two bytes; split it down the middle.
        let payload = "data: {\"text\":\"α\"}\n\n".as_bytes().to_vec();
        let split = payload.iter().position(|&b| b == 0xCE).unwrap() + 1;
        let mut d = SseDecoder::new();
        assert!(d.push(&payload[..split]).is_empty());
        let out = d.push(&payload[split..]);
        assert_eq!(out, vec!["{\"text\":\"α\"}"]);
    }

    #[test]
    fn skips_comment_heartbeats() {
        let mut d = SseDecoder::new();
        let out = d.push(b": ping\n\ndata: {\"a\":1}\n\n");
        assert_eq!(out, vec![r#"{"a":1}"#]);
    }

    #[test]
    fn joins_multiple_data_lines() {
        let mut d = SseDecoder::new();
        let out = d.push(b"data: line1\ndata: line2\n\n");
        assert_eq!(out, vec!["line1\nline2"]);
    }

    #[test]
    fn handles_crlf_line_endings() {
        let mut d = SseDecoder::new();
        let out = d.push(b"event: x\r\ndata: {\"a\":1}\r\n\r\n");
        assert_eq!(out, vec![r#"{"a":1}"#]);
    }

    // --- TurnAccumulator ---------------------------------------------------

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

    #[test]
    fn accumulates_a_text_turn() {
        let body = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-5\",\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":900}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello \"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ).as_bytes();

        let (events, turn) = run(body);
        assert_eq!(turn.text(), "Hello world");
        assert_eq!(turn.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(turn.model, "claude-opus-5");
        assert_eq!(turn.usage.output_tokens, 7);
        // message_delta.usage must not clobber input counts from message_start.
        assert_eq!(turn.usage.input_tokens, 10);
        assert_eq!(turn.usage.cache_read_input_tokens, 900);
        assert_eq!(events.last(), Some(&TurnEvent::Done));
        assert!(events.contains(&TurnEvent::TextDelta("Hello ".to_string())));
    }

    #[test]
    fn accumulates_a_tool_use_turn_from_json_fragments() {
        let body = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"run_command\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"commands\\\":\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"load 1crn\\\"}\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ).as_bytes();

        let (events, turn) = run(body);
        assert_eq!(turn.stop_reason, Some(StopReason::ToolUse));
        let uses = turn.tool_uses();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].0, "toolu_1");
        assert_eq!(uses[0].1, "run_command");
        assert_eq!(uses[0].2["commands"], "load 1crn");
        assert!(events.contains(&TurnEvent::ToolUseStarted {
            name: "run_command".to_string()
        }));
    }

    #[test]
    fn preserves_thinking_blocks_with_omitted_content() {
        // claude-opus-5 defaults to display "omitted": empty text, real signature.
        let body = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-xyz\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ).as_bytes();

        let (events, turn) = run(body);
        assert!(events.contains(&TurnEvent::ThinkingStarted));
        assert_eq!(turn.content.len(), 2);
        assert_eq!(
            turn.content[0],
            ContentBlock::Thinking {
                thinking: String::new(),
                signature: "sig-xyz".to_string(),
            }
        );
        assert_eq!(turn.text(), "done");
    }

    #[test]
    fn captures_refusal_stop_reason_and_details() {
        let body = concat!(
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_details\":{\"category\":\"cyber\",\"explanation\":\"declined\"}}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        assert_eq!(turn.stop_reason, Some(StopReason::Refusal));
        // Content is empty on a pre-output refusal — indexing content[0] here
        // is exactly the bug this guards against.
        assert!(turn.content.is_empty());
        assert_eq!(turn.stop_details.unwrap().category.as_deref(), Some("cyber"));
    }

    #[test]
    fn surfaces_mid_stream_errors() {
        let body =
            b"data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"overloaded\"}}\n\n";
        let (events, _) = run(body);
        assert_eq!(events, vec![TurnEvent::Error("overloaded".to_string())]);
    }

    #[test]
    fn ignores_unknown_events_and_deltas() {
        let body = concat!(
            "data: {\"type\":\"ping\"}\n\n",
            "data: {\"type\":\"some_future_event\",\"index\":0}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"future_delta\",\"x\":1}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        assert_eq!(turn.text(), "ok");
    }

    #[test]
    fn malformed_tool_input_degrades_to_empty_object() {
        let body = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"screenshot\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{oops\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        assert_eq!(*turn.tool_uses()[0].2, serde_json::json!({}));
    }

    #[test]
    fn tool_use_with_no_input_deltas_yields_empty_object() {
        let body = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"get_scene_state\",\"input\":{}}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        assert_eq!(*turn.tool_uses()[0].2, serde_json::json!({}));
    }

    #[test]
    fn interleaved_block_indices_are_kept_in_order() {
        let body = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"a\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"first\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ).as_bytes();

        let (_, turn) = run(body);
        assert_eq!(turn.content.len(), 2);
        assert!(matches!(turn.content[0], ContentBlock::Text { .. }));
        assert!(matches!(turn.content[1], ContentBlock::ToolUse { .. }));
    }
}
