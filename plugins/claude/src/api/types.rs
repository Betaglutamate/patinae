//! Wire types for `POST /v1/messages`.
//!
//! Hand-written because there is no official Anthropic Rust SDK. Field names
//! match the JSON exactly; anything optional is skipped when absent so request
//! bodies stay byte-stable, which matters for prompt caching.
//!
//! Deliberately absent: `temperature`, `top_p`, `top_k`, and
//! `thinking.budget_tokens`. All four are rejected with a 400 on `claude-opus-5`.

use serde::{Deserialize, Serialize};

/// Default model. Thinking is on by default on this model, so no `thinking`
/// field is sent at all.
pub const DEFAULT_MODEL: &str = "claude-opus-5";

// =============================================================================
// Content blocks
// =============================================================================

/// Source of an image block. Only base64 is used — screenshots are produced
/// locally, never fetched from a URL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
}

impl ImageSource {
    pub fn png(base64_data: impl Into<String>) -> Self {
        ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: base64_data.into(),
        }
    }
}

/// A block inside a `tool_result`. Kept separate from [`ContentBlock`] because
/// only text and images are legal here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image { source: ImageSource },
}

impl ToolResultContent {
    pub fn text(text: impl Into<String>) -> Self {
        ToolResultContent::Text { text: text.into() }
    }

    pub fn png(base64_data: impl Into<String>) -> Self {
        ToolResultContent::Image {
            source: ImageSource::png(base64_data),
        }
    }
}

/// A content block in either direction.
///
/// `Thinking` and `RedactedThinking` are echoed back to the API **unchanged**
/// when continuing a conversation on the same model; editing or reconstructing
/// them causes the turn to be rejected. On `claude-opus-5` the `thinking` text
/// is an empty string by default (`display` defaults to `"omitted"`), but the
/// signature still matters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        is_error: Option<bool>,
    },
    Image {
        source: ImageSource,
    },
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text { text: text.into() }
    }

    /// A successful tool result.
    pub fn tool_ok(tool_use_id: impl Into<String>, content: Vec<ToolResultContent>) -> Self {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.into(),
            content,
            is_error: None,
        }
    }

    /// A failed tool result. Claude handles these gracefully and adapts, so
    /// prefer this over silently dropping the call.
    pub fn tool_err(tool_use_id: impl Into<String>, message: impl Into<String>) -> Self {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: vec![ToolResultContent::text(message)],
            is_error: Some(true),
        }
    }
}

// =============================================================================
// Messages
// =============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content,
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::user(vec![ContentBlock::text(text)])
    }
}

// =============================================================================
// Request
// =============================================================================

/// Marks a prefix as cacheable. Placed on the last system block so the whole
/// system prompt (several thousand tokens of Patinae knowledge) is served from
/// cache at ~0.1x on every turn after the first.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: String,
}

impl Default for CacheControl {
    fn default() -> Self {
        Self {
            kind: "ephemeral".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    /// A system block marked as the end of the cacheable prefix.
    pub fn cached(text: impl Into<String>) -> Self {
        Self {
            kind: "text".to_string(),
            text: text.into(),
            cache_control: Some(CacheControl::default()),
        }
    }
}

/// A tool Claude may call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub system: Vec<SystemBlock>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    pub stream: bool,
}

// =============================================================================
// Response
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    PauseTurn,
    Refusal,
    /// Forward-compatibility: a stop reason introduced after this was written.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

/// Structured detail attached to a refusal. Present only when
/// `stop_reason == Refusal`, and `category` may still be null.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct StopDetails {
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub explanation: Option<String>,
}

/// A fully accumulated assistant turn.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssistantTurn {
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<StopReason>,
    pub stop_details: Option<StopDetails>,
    pub usage: Usage,
    pub model: String,
}

impl AssistantTurn {
    /// Concatenated text across all text blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Every `tool_use` block, in order. Each one needs a matching
    /// `tool_result` in the follow-up user message.
    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.as_str(), name.as_str(), input))
                }
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_omits_sampling_and_thinking_params() {
        let req = MessagesRequest {
            model: DEFAULT_MODEL.to_string(),
            max_tokens: 64000,
            messages: vec![Message::user_text("hi")],
            system: vec![SystemBlock::cached("you are a viewer agent")],
            tools: vec![],
            output_config: Some(OutputConfig {
                effort: Some("xhigh".to_string()),
            }),
            stream: true,
        };
        let json = serde_json::to_string(&req).unwrap();

        // These four are 400s on claude-opus-5.
        for forbidden in ["temperature", "top_p", "top_k", "budget_tokens"] {
            assert!(!json.contains(forbidden), "request leaked {forbidden}");
        }
        assert!(json.contains(r#""effort":"xhigh""#));
        assert!(json.contains(r#""cache_control":{"type":"ephemeral"}"#));
    }

    #[test]
    fn empty_tools_and_system_are_omitted() {
        let req = MessagesRequest {
            model: DEFAULT_MODEL.to_string(),
            max_tokens: 1024,
            messages: vec![Message::user_text("hi")],
            system: vec![],
            tools: vec![],
            output_config: None,
            stream: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("\"tools\""));
        assert!(!json.contains("\"system\""));
        assert!(!json.contains("\"output_config\""));
    }

    #[test]
    fn tool_result_success_omits_is_error() {
        let block = ContentBlock::tool_ok("toolu_1", vec![ToolResultContent::text("done")]);
        let json = serde_json::to_string(&block).unwrap();
        assert!(!json.contains("is_error"));
        assert!(json.contains(r#""tool_use_id":"toolu_1""#));
    }

    #[test]
    fn tool_result_error_sets_is_error() {
        let block = ContentBlock::tool_err("toolu_1", "denied by user");
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""is_error":true"#));
        assert!(json.contains("denied by user"));
    }

    #[test]
    fn image_blocks_serialize_as_base64_source() {
        let block = ContentBlock::tool_ok("toolu_1", vec![ToolResultContent::png("QUJD")]);
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"image""#));
        assert!(json.contains(r#""media_type":"image/png""#));
        assert!(json.contains(r#""data":"QUJD""#));
    }

    #[test]
    fn thinking_blocks_round_trip_unchanged() {
        // Must survive a round trip byte-for-byte: the API rejects edited
        // thinking blocks when continuing on the same model.
        let original = ContentBlock::Thinking {
            thinking: String::new(),
            signature: "sig-abc".to_string(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn unknown_stop_reason_does_not_fail_parsing() {
        let r: StopReason = serde_json::from_str(r#""some_future_reason""#).unwrap();
        assert_eq!(r, StopReason::Unknown);
    }

    #[test]
    fn known_stop_reasons_parse() {
        let cases = [
            (r#""tool_use""#, StopReason::ToolUse),
            (r#""end_turn""#, StopReason::EndTurn),
            (r#""refusal""#, StopReason::Refusal),
            (r#""pause_turn""#, StopReason::PauseTurn),
            (r#""max_tokens""#, StopReason::MaxTokens),
        ];
        for (json, expected) in cases {
            assert_eq!(serde_json::from_str::<StopReason>(json).unwrap(), expected);
        }
    }

    #[test]
    fn assistant_turn_extracts_text_and_tool_uses() {
        let turn = AssistantTurn {
            content: vec![
                ContentBlock::text("Loading "),
                ContentBlock::text("the structure."),
                ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "run_command".to_string(),
                    input: serde_json::json!({"commands": "load 1crn"}),
                },
            ],
            ..Default::default()
        };
        assert_eq!(turn.text(), "Loading the structure.");
        let uses = turn.tool_uses();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].1, "run_command");
    }
}
