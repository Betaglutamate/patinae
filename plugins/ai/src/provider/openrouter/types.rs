//! Wire types for OpenRouter's `POST /api/v1/chat/completions`.
//!
//! OpenRouter speaks the OpenAI Chat Completions dialect, plus a small set of
//! its own fields (`reasoning`, `reasoning_details`, `provider`). The shape is
//! flatter and lossier than either Anthropic's or Google's, and two of its
//! constraints drive real decisions in the adapter above:
//!
//! - **A `tool` message's content is a plain string.** There is no content-part
//!   array, so a tool result cannot carry an image.
//! - **Tool arguments are a JSON string, not a JSON value**, and stream in as
//!   fragments of that string keyed by an index rather than by an id.

use serde::{Deserialize, Serialize};

/// Who authored a message.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A piece of a user message.
///
/// Only user messages take an array of these; system, assistant and tool
/// messages take a bare string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    /// Images travel as a data URI in `image_url.url` — the same field used for
    /// remote images, which is why the payload is a URI rather than raw base64.
    ImageUrl {
        #[serde(rename = "image_url")]
        image_url: ImageUrl,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageUrl {
    pub url: String,
}

impl ImageUrl {
    pub fn data_uri(mime: &str, base64_data: &str) -> Self {
        Self {
            url: format!("data:{mime};base64,{base64_data}"),
        }
    }
}

/// Message content, which the API accepts as either a string or an array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    /// The arguments, as a JSON *string*. Models emit invalid JSON here often
    /// enough that it must be parsed defensively rather than trusted.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

impl ToolCall {
    pub fn function(id: impl Into<String>, name: impl Into<String>, arguments: String) -> Self {
        Self {
            id: id.into(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Message {
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(rename = "tool_calls", skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(rename = "tool_call_id", skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// OpenRouter's normalised reasoning trace. Replayed verbatim so upstream
    /// vendors that validate their own signatures keep accepting the history.
    #[serde(rename = "reasoning_details", skip_serializing_if = "Option::is_none")]
    pub reasoning_details: Option<Vec<serde_json::Value>>,
}

impl Message {
    pub fn new(role: Role, content: Content) -> Self {
        Self {
            role: Some(role),
            content: Some(content),
            ..Default::default()
        }
    }

    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self::new(role, Content::Text(text.into()))
    }

    /// A tool result. Content is a string here because the API allows nothing
    /// else in this position.
    pub fn tool_result(tool_call_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: Some(Role::Tool),
            content: Some(Content::Text(text.into())),
            tool_call_id: Some(tool_call_id.into()),
            ..Default::default()
        }
    }
}

// =============================================================================
// Request
// =============================================================================

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

/// Reasoning controls, normalised by OpenRouter across upstream vendors.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Reasoning {
    /// `"low"`, `"medium"` or `"high"` — the only three the API defines.
    pub effort: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    pub max_tokens: u32,
    pub stream: bool,
}

// =============================================================================
// Response
// =============================================================================

/// A tool call as it streams in.
///
/// Every field but `index` is optional: the first fragment carries the id and
/// name, and later fragments extend `arguments` alone.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ToolCallDelta {
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionDelta>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct FunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct Delta {
    #[serde(default)]
    pub content: Option<String>,
    /// Human-readable reasoning text, when the upstream model exposes it.
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_details: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct Choice {
    #[serde(default)]
    pub delta: Option<Delta>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// OpenRouter's own field, set when the upstream provider errored after the
    /// stream had already started.
    #[serde(default)]
    pub error: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct ChatChunk {
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub model: Option<String>,
}

/// A fully accumulated assistant turn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssistantTurn {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub reasoning_details: Vec<serde_json::Value>,
    pub finish_reason: Option<String>,
    pub usage: Usage,
    pub model: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn user_content_serializes_as_a_string_or_an_array() {
        let plain = serde_json::to_value(Message::text(Role::User, "hi")).unwrap();
        assert_eq!(plain["content"], "hi");

        let mixed = serde_json::to_value(Message::new(
            Role::User,
            Content::Parts(vec![
                ContentPart::Text {
                    text: "look".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl::data_uri("image/png", "QUJD"),
                },
            ]),
        ))
        .unwrap();
        assert!(mixed["content"].is_array());
        assert_eq!(mixed["content"][1]["type"], "image_url");
    }

    #[test]
    fn images_travel_as_a_data_uri() {
        assert_eq!(
            ImageUrl::data_uri("image/png", "QUJD").url,
            "data:image/png;base64,QUJD"
        );
    }

    #[test]
    fn a_tool_result_carries_its_call_id_and_omits_empty_fields() {
        let json = serde_json::to_value(Message::tool_result("call_1", "done")).unwrap();
        assert_eq!(json["role"], "tool");
        assert_eq!(json["tool_call_id"], "call_1");
        assert_eq!(json["content"], "done");
        assert!(json.get("tool_calls").is_none());
        assert!(json.get("reasoning_details").is_none());
    }

    #[test]
    fn a_streaming_tool_call_delta_parses_with_only_an_index() {
        // Later fragments carry neither id nor name.
        let d: ToolCallDelta =
            serde_json::from_value(json!({"index": 0, "function": {"arguments": "1crn\"}"}}))
                .unwrap();
        assert_eq!(d.index, 0);
        assert!(d.id.is_none());
        assert_eq!(d.function.unwrap().arguments.as_deref(), Some("1crn\"}"));
    }

    #[test]
    fn a_chunk_parses_with_unknown_fields_present() {
        // Real chunks carry id, object, created, system_fingerprint, provider …
        let chunk: ChatChunk = serde_json::from_value(json!({
            "id": "gen-123",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000u64,
            "provider": "Anthropic",
            "model": "anthropic/claude-sonnet-5",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}],
        }))
        .unwrap();
        assert_eq!(chunk.model.as_deref(), Some("anthropic/claude-sonnet-5"));
        assert_eq!(
            chunk.choices[0].delta.as_ref().unwrap().content.as_deref(),
            Some("hi")
        );
    }

    #[test]
    fn the_request_omits_optional_blocks_when_they_are_empty() {
        let req = ChatRequest {
            model: "anthropic/claude-sonnet-5".into(),
            messages: vec![Message::text(Role::User, "hi")],
            tools: vec![],
            reasoning: None,
            max_tokens: 1024,
            stream: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("\"tools\""));
        assert!(!json.contains("\"reasoning\""));
    }
}
