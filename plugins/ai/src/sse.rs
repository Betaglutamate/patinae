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
}
