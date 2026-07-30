//! Wire types for `:streamGenerateContent` on the Gemini Developer API.
//!
//! Hand-written because Google ships no Gemini Rust SDK. Field names match the
//! JSON exactly — the API is proto-derived, so it is `camelCase` throughout and
//! strict about unknown fields, which is why nothing optional is ever sent as
//! `null`.

use serde::{Deserialize, Serialize};

/// Whose turn a `Content` belongs to.
///
/// Only two values exist. Tool results are *not* a third role: they are
/// `functionResponse` parts inside a `user` turn, which is the shape the REST
/// API accepts even though several SDKs expose a synthetic `function` role.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Model,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Blob {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

/// The result of a tool call.
///
/// `response` must be a JSON *object*; a bare string is rejected. Text results
/// are therefore wrapped as `{"result": "…"}` on the way in.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

/// One piece of a turn.
///
/// Untagged rather than internally tagged: the API discriminates by *which*
/// field is present, not by a `type` field, so `{"text": …}` and
/// `{"functionCall": …}` are distinguished structurally. Order matters —
/// `Text` is last because it is the most permissive variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Part {
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: Blob,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCall,
        /// Present when the call was produced with thinking enabled. Must be
        /// echoed back unchanged or the model loses its own chain of thought.
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponse,
    },
    Text {
        text: String,
        /// `true` marks the text as the model's private reasoning rather than
        /// its answer. Absent on ordinary prose, so it is skipped when false.
        #[serde(skip_serializing_if = "Option::is_none")]
        thought: Option<bool>,
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
}

impl Part {
    pub fn text(text: impl Into<String>) -> Self {
        Part::Text {
            text: text.into(),
            thought: None,
            thought_signature: None,
        }
    }

    pub fn is_thought(&self) -> bool {
        matches!(
            self,
            Part::Text {
                thought: Some(true),
                ..
            }
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    pub parts: Vec<Part>,
}

impl Content {
    pub fn new(role: Role, parts: Vec<Part>) -> Self {
        Self {
            role: Some(role),
            parts,
        }
    }
}

// =============================================================================
// Request
// =============================================================================

/// A tool the model may call.
///
/// `parameters` is omitted entirely for a tool that takes no arguments: the API
/// rejects a declaration carrying an empty `properties` object.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Tool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<FunctionDeclaration>,
}

/// How much the model may think before answering.
///
/// `thinkingBudget` is a token allowance, not a level: `-1` asks the model to
/// decide for itself, and `0` disables thinking on models that permit it.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct ThinkingConfig {
    #[serde(rename = "includeThoughts")]
    pub include_thoughts: bool,
    #[serde(rename = "thinkingBudget")]
    pub thinking_budget: i32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GenerationConfig {
    #[serde(rename = "maxOutputTokens")]
    pub max_output_tokens: u32,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(rename = "generationConfig")]
    pub generation_config: GenerationConfig,
}

// =============================================================================
// Response
// =============================================================================

/// Why generation stopped.
///
/// The safety-adjacent variants are all distinct reasons the model declined,
/// and are collapsed into one neutral refusal above this layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    Stop,
    MaxTokens,
    Safety,
    Recitation,
    Language,
    Blocklist,
    ProhibitedContent,
    Spii,
    MalformedFunctionCall,
    ImageSafety,
    UnexpectedToolCall,
    Other,
    /// Forward compatibility: a reason introduced after this was written.
    #[serde(other)]
    Unknown,
}

impl FinishReason {
    /// Whether this is the model declining on policy grounds.
    pub fn is_refusal(self) -> bool {
        matches!(
            self,
            FinishReason::Safety
                | FinishReason::Recitation
                | FinishReason::Blocklist
                | FinishReason::ProhibitedContent
                | FinishReason::Spii
                | FinishReason::ImageSafety
        )
    }
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub struct UsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    pub prompt_tokens: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    pub candidates_tokens: u32,
    #[serde(rename = "cachedContentTokenCount", default)]
    pub cached_tokens: u32,
    #[serde(rename = "thoughtsTokenCount", default)]
    pub thoughts_tokens: u32,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Candidate {
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(rename = "finishReason", default)]
    pub finish_reason: Option<FinishReason>,
}

/// Set when the *prompt* was blocked, in which case no candidate is returned at
/// all — the reason for the refusal lives here instead.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct PromptFeedback {
    #[serde(rename = "blockReason", default)]
    pub block_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata", default)]
    pub usage_metadata: Option<UsageMetadata>,
    #[serde(rename = "modelVersion", default)]
    pub model_version: Option<String>,
    #[serde(rename = "promptFeedback", default)]
    pub prompt_feedback: Option<PromptFeedback>,
}

/// A fully accumulated assistant turn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssistantTurn {
    pub parts: Vec<Part>,
    pub finish_reason: Option<FinishReason>,
    pub block_reason: Option<String>,
    pub usage: UsageMetadata,
    pub model: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parts_are_discriminated_structurally_not_by_a_type_tag() {
        let cases = [
            (json!({"text": "hi"}), "text"),
            (json!({"functionCall": {"name": "f", "args": {}}}), "call"),
            (
                json!({"functionResponse": {"name": "f", "response": {"result": "ok"}}}),
                "response",
            ),
            (
                json!({"inlineData": {"mimeType": "image/png", "data": "QUJD"}}),
                "blob",
            ),
        ];
        for (value, what) in cases {
            let part: Part = serde_json::from_value(value).unwrap_or_else(|e| {
                panic!("{what} should deserialize: {e}");
            });
            match (what, &part) {
                ("text", Part::Text { .. })
                | ("call", Part::FunctionCall { .. })
                | ("response", Part::FunctionResponse { .. })
                | ("blob", Part::InlineData { .. }) => {}
                _ => panic!("{what} deserialized to {part:?}"),
            }
        }
    }

    #[test]
    fn ordinary_text_omits_the_thought_fields() {
        // An explicit `"thought": null` is an unknown-shaped field to a
        // proto-derived API; absence is the only safe encoding.
        let json = serde_json::to_string(&Part::text("hello")).unwrap();
        assert_eq!(json, r#"{"text":"hello"}"#);
    }

    #[test]
    fn thought_parts_round_trip_with_their_signature() {
        let original = Part::Text {
            text: String::new(),
            thought: Some(true),
            thought_signature: Some("sig-abc".into()),
        };
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(r#""thoughtSignature":"sig-abc""#));
        assert_eq!(serde_json::from_str::<Part>(&json).unwrap(), original);
        assert!(original.is_thought());
    }

    #[test]
    fn function_calls_carry_a_thought_signature_when_one_was_issued() {
        let original = Part::FunctionCall {
            function_call: FunctionCall {
                name: "run_command".into(),
                args: json!({"commands": "load 1crn"}),
            },
            thought_signature: Some("sig".into()),
        };
        let back: Part = serde_json::from_str(&serde_json::to_string(&original).unwrap()).unwrap();
        assert_eq!(back, original);
    }

    #[test]
    fn a_tool_with_no_arguments_omits_parameters_entirely() {
        // The API rejects a declaration whose `properties` is an empty object.
        let decl = FunctionDeclaration {
            name: "get_scene_state".into(),
            description: "d".into(),
            parameters: None,
        };
        assert!(!serde_json::to_string(&decl).unwrap().contains("parameters"));
    }

    #[test]
    fn known_finish_reasons_parse_and_refusals_are_recognised() {
        let cases = [
            (r#""STOP""#, FinishReason::Stop, false),
            (r#""MAX_TOKENS""#, FinishReason::MaxTokens, false),
            (r#""SAFETY""#, FinishReason::Safety, true),
            (
                r#""PROHIBITED_CONTENT""#,
                FinishReason::ProhibitedContent,
                true,
            ),
            (r#""SPII""#, FinishReason::Spii, true),
            (r#""BLOCKLIST""#, FinishReason::Blocklist, true),
            (
                r#""MALFORMED_FUNCTION_CALL""#,
                FinishReason::MalformedFunctionCall,
                false,
            ),
        ];
        for (json, expected, refusal) in cases {
            let parsed: FinishReason = serde_json::from_str(json).unwrap();
            assert_eq!(parsed, expected, "{json}");
            assert_eq!(parsed.is_refusal(), refusal, "{json}");
        }
    }

    #[test]
    fn an_unknown_finish_reason_does_not_fail_the_turn() {
        let r: FinishReason = serde_json::from_str(r#""SOME_FUTURE_REASON""#).unwrap();
        assert_eq!(r, FinishReason::Unknown);
    }

    #[test]
    fn a_blocked_prompt_parses_with_no_candidates() {
        // Indexing candidates[0] here is exactly the bug this guards against.
        let body = json!({"promptFeedback": {"blockReason": "SAFETY"}});
        let parsed: GenerateContentResponse = serde_json::from_value(body).unwrap();
        assert!(parsed.candidates.is_empty());
        assert_eq!(
            parsed.prompt_feedback.unwrap().block_reason.as_deref(),
            Some("SAFETY")
        );
    }

    #[test]
    fn a_response_chunk_parses_with_unknown_fields_present() {
        // Real chunks carry safetyRatings, avgLogprobs, citationMetadata …
        let body = json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hi"}]},
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"}],
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "thoughtsTokenCount": 11},
            "modelVersion": "gemini-2.5-pro",
        });
        let parsed: GenerateContentResponse = serde_json::from_value(body).unwrap();
        assert_eq!(parsed.candidates[0].finish_reason, Some(FinishReason::Stop));
        assert_eq!(parsed.usage_metadata.unwrap().thoughts_tokens, 11);
    }
}
