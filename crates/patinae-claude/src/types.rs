//! Messages API wire types.
//!
//! Only the subset Patinae needs is modelled. Unknown content blocks and stop
//! reasons deserialize into catch-all variants rather than failing, so a
//! server-side addition degrades to "we don't render it" instead of breaking
//! every request.

use serde::{Deserialize, Serialize};

/// Anthropic API version header value.
pub const API_VERSION: &str = "2023-06-01";

/// Beta flag enabling the `fallbacks: "default"` request field.
pub const BETA_SERVER_SIDE_FALLBACK: &str = "server-side-fallback-2026-07-01";

/// Messages endpoint.
pub const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

/// Who authored a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// The user, or a turn carrying tool results.
    User,
    /// The model.
    Assistant,
}

/// Base64 image payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSource {
    /// Always `base64` for the payloads Patinae sends.
    #[serde(rename = "type")]
    pub source_type: String,
    /// MIME type, e.g. `image/png`.
    pub media_type: String,
    /// Base64-encoded bytes, no newlines.
    pub data: String,
}

impl ImageSource {
    /// Builds a base64 PNG source.
    pub fn png(data: impl Into<String>) -> Self {
        Self {
            source_type: "base64".to_string(),
            media_type: "image/png".to_string(),
            data: data.into(),
        }
    }
}

/// Cache breakpoint marker.
///
/// Owned rather than `&'static str` because this type also appears on the
/// response side, and borrowed fields cannot be deserialized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheControl {
    /// Always `ephemeral`.
    #[serde(rename = "type")]
    pub kind: String,
}

impl CacheControl {
    /// Returns the default 5-minute ephemeral breakpoint.
    pub fn ephemeral() -> Self {
        Self {
            kind: "ephemeral".to_string(),
        }
    }
}

/// A block of message content.
///
/// [`ContentBlock::Unknown`] preserves anything this build does not recognise
/// so it can still be echoed back to the API verbatim, which matters for
/// thinking-adjacent blocks whose exact shape must round-trip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text.
    Text {
        /// The text.
        text: String,
        /// Optional cache breakpoint.
        #[serde(skip_serializing_if = "Option::is_none", default)]
        cache_control: Option<CacheControl>,
    },
    /// Summarized reasoning. Must be echoed back unchanged on the same model.
    Thinking {
        /// Reasoning text. Empty unless `display` is `summarized`.
        thinking: String,
        /// Opaque signature; the API rejects tampering.
        #[serde(skip_serializing_if = "Option::is_none", default)]
        signature: Option<String>,
    },
    /// Encrypted reasoning. Opaque; echo back unchanged.
    RedactedThinking {
        /// Opaque payload.
        data: String,
    },
    /// A tool invocation requested by the model.
    ToolUse {
        /// Correlation id echoed back on the matching result.
        id: String,
        /// Tool name.
        name: String,
        /// Parsed tool input.
        input: serde_json::Value,
    },
    /// The result of a tool invocation.
    ToolResult {
        /// Id of the `tool_use` block being answered.
        tool_use_id: String,
        /// Result payload.
        content: Vec<ContentBlock>,
        /// Whether the tool failed.
        #[serde(skip_serializing_if = "std::ops::Not::not", default)]
        is_error: bool,
    },
    /// An image.
    Image {
        /// Image payload.
        source: ImageSource,
    },
    /// A block type this build does not model.
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

impl ContentBlock {
    /// Builds a plain text block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    /// Builds a text block that closes a cacheable prefix.
    pub fn cached_text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: Some(CacheControl::ephemeral()),
        }
    }

    /// Builds a successful tool result carrying text.
    pub fn tool_result(tool_use_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: vec![Self::text(text)],
            is_error: false,
        }
    }

    /// Builds a failed tool result.
    pub fn tool_error(tool_use_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: vec![Self::text(text)],
            is_error: true,
        }
    }

    /// Returns the text of a [`ContentBlock::Text`] block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text, .. } => Some(text),
            _ => None,
        }
    }
}

/// One turn of the conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Turn author.
    pub role: Role,
    /// Turn content.
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Builds a user turn from plain text.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Builds a user turn from arbitrary blocks (tool results, images).
    pub fn user_blocks(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }

    /// Builds an assistant turn.
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content,
        }
    }
}

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Finished naturally.
    EndTurn,
    /// Hit `max_tokens`.
    MaxTokens,
    /// Hit a stop sequence.
    StopSequence,
    /// Wants one or more tools executed.
    ToolUse,
    /// Server-side tool loop paused; resend to continue.
    PauseTurn,
    /// Declined by safety classifiers.
    Refusal,
    /// A stop reason this build does not model.
    #[serde(untagged)]
    Unknown(String),
}

/// Structured detail attached to a refusal.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StopDetails {
    /// Policy category, when the API supplies one.
    #[serde(default)]
    pub category: Option<String>,
    /// Human-readable explanation, when the API supplies one.
    #[serde(default)]
    pub explanation: Option<String>,
}

/// Token accounting for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Uncached input tokens.
    #[serde(default)]
    pub input_tokens: u64,
    /// Generated tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Tokens written to the cache this request.
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    /// Tokens served from the cache this request.
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

impl Usage {
    /// Total prompt size, cached and uncached.
    ///
    /// `input_tokens` alone is only the uncached remainder, so reading it as
    /// "prompt size" badly under-reports a warm conversation.
    pub fn total_input_tokens(&self) -> u64 {
        self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens
    }
}

/// Reasoning configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ThinkingConfig {
    /// Always `adaptive` for the models Patinae targets.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// `summarized` to surface reasoning, `omitted` to hide it.
    pub display: &'static str,
}

impl ThinkingConfig {
    /// Adaptive thinking with a readable summary.
    pub fn summarized() -> Self {
        Self {
            kind: "adaptive",
            display: "summarized",
        }
    }
}

/// Output shaping options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OutputConfig {
    /// Reasoning depth and overall token spend.
    pub effort: &'static str,
}

/// A client-side tool definition sent to the model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// What the tool does and, importantly, *when* to call it.
    pub description: String,
    /// JSON Schema for the tool input.
    pub input_schema: serde_json::Value,
    /// Whether inputs are guaranteed to validate against the schema.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub strict: bool,
}

/// An Anthropic-hosted tool that runs server-side.
///
/// These carry a versioned `type` and no input schema, so they cannot share a
/// struct with [`ToolDefinition`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ServerTool {
    /// Versioned tool type, e.g. `web_search_20260209`.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Tool name, e.g. `web_search`.
    pub name: &'static str,
}

/// Either kind of tool, serialized flat into the `tools` array.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Tool {
    /// A tool Patinae executes.
    Custom(ToolDefinition),
    /// A tool Anthropic executes.
    Server(ServerTool),
}

/// A request to the Messages API.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MessageRequest {
    /// Model identifier.
    pub model: String,
    /// Hard ceiling on thinking plus response tokens.
    pub max_tokens: u32,
    /// Always `true`; large `max_tokens` needs streaming to avoid timeouts.
    pub stream: bool,
    /// System prompt blocks. The last one carries the cache breakpoint.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub system: Vec<ContentBlock>,
    /// Conversation so far.
    pub messages: Vec<Message>,
    /// Available tools.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    /// Reasoning configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    /// Output shaping.
    pub output_config: OutputConfig,
    /// Server-side refusal fallback. `"default"` lets the API pick.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallbacks: Option<&'static str>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_content_blocks_round_trip() {
        let blocks = vec![
            ContentBlock::text("hello"),
            ContentBlock::cached_text("cached"),
            ContentBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "run_command".to_string(),
                input: serde_json::json!({"command": "zoom"}),
            },
            ContentBlock::tool_result("toolu_1", "done"),
            ContentBlock::tool_error("toolu_2", "denied"),
            ContentBlock::Image {
                source: ImageSource::png("QUJD"),
            },
            ContentBlock::Thinking {
                thinking: String::new(),
                signature: Some("sig".to_string()),
            },
        ];

        let json = serde_json::to_string(&blocks).expect("blocks should serialize");
        let restored: Vec<ContentBlock> =
            serde_json::from_str(&json).expect("blocks should deserialize");

        assert_eq!(restored, blocks);
    }

    #[test]
    fn unknown_content_blocks_survive_a_round_trip() {
        // A block type added server-side after this build shipped.
        let raw = r#"[{"type":"future_block","payload":{"a":1}}]"#;

        let blocks: Vec<ContentBlock> =
            serde_json::from_str(raw).expect("unknown blocks should not fail parsing");

        assert!(matches!(blocks[0], ContentBlock::Unknown(_)));
        let reserialized = serde_json::to_string(&blocks).expect("unknown blocks should serialize");
        assert!(reserialized.contains("future_block"));
        assert!(reserialized.contains("\"a\":1"));
    }

    #[test]
    fn unknown_stop_reasons_do_not_fail_parsing() {
        let reason: StopReason =
            serde_json::from_str("\"some_new_reason\"").expect("unknown reason should parse");
        assert_eq!(reason, StopReason::Unknown("some_new_reason".to_string()));

        let known: StopReason = serde_json::from_str("\"tool_use\"").expect("known reason parses");
        assert_eq!(known, StopReason::ToolUse);
    }

    #[test]
    fn is_error_is_omitted_when_false() {
        let ok = serde_json::to_string(&ContentBlock::tool_result("t", "fine"))
            .expect("block should serialize");
        assert!(!ok.contains("is_error"));

        let bad = serde_json::to_string(&ContentBlock::tool_error("t", "bad"))
            .expect("block should serialize");
        assert!(bad.contains("\"is_error\":true"));
    }

    #[test]
    fn total_input_tokens_counts_cached_and_uncached() {
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
            cache_creation_input_tokens: 100,
            cache_read_input_tokens: 1000,
        };

        // Reading `input_tokens` alone would report 10 for a 1110-token prompt.
        assert_eq!(usage.total_input_tokens(), 1110);
    }

    #[test]
    fn request_omits_empty_optional_sections() {
        let request = MessageRequest {
            model: "claude-opus-5".to_string(),
            max_tokens: 32_000,
            stream: true,
            system: Vec::new(),
            messages: vec![Message::user("hi")],
            tools: Vec::new(),
            thinking: None,
            output_config: OutputConfig { effort: "high" },
            fallbacks: None,
        };

        let json = serde_json::to_string(&request).expect("request should serialize");

        assert!(!json.contains("\"system\""));
        assert!(!json.contains("\"tools\""));
        assert!(!json.contains("\"thinking\""));
        assert!(!json.contains("\"fallbacks\""));
        assert!(json.contains("\"stream\":true"));
    }
}
