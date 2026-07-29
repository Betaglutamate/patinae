//! The provider seam.
//!
//! Everything above this module — the agent loop, tools, panel, transcript,
//! prompt — is vendor-neutral. Everything below it translates to and from one
//! vendor's wire format. Neither Anthropic nor Google ships a Rust SDK, so both
//! sides are hand-written.
//!
//! The neutral conversation model is deliberately *not* either vendor's shape.
//! Two details make it work across both:
//!
//! - **Tool calls carry both an `id` and a `name`.** Anthropic pairs results to
//!   calls by id; Gemini's `functionResponse` wants the name as well. Carrying
//!   both means neither provider has to invent one.
//! - **Reasoning is opaque and provider-tagged.** Anthropic requires thinking
//!   blocks to be echoed back byte-identically, and Gemini has its own
//!   `thoughtSignature`. Neither is meaningful to the other, so [`Part::Reasoning`]
//!   records which provider produced it and [`sanitize_history`] drops foreign
//!   ones when the user switches. Both APIs tolerate their absence.

pub mod anthropic;

use serde_json::Value;

use crate::auth::AuthBroker;

/// Which vendor backs the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderId {
    Claude,
    Gemini,
}

impl ProviderId {
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderId::Claude => "claude",
            ProviderId::Gemini => "gemini",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ProviderId::Claude => "Claude",
            ProviderId::Gemini => "Gemini",
        }
    }

    /// Model used when the per-provider model setting is unset.
    ///
    /// Per-provider rather than one global default: the two vendors' model names
    /// share no namespace, and users switch back and forth.
    pub fn default_model(self) -> &'static str {
        match self {
            ProviderId::Claude => anthropic::types::DEFAULT_MODEL,
            ProviderId::Gemini => "gemini-2.5-pro",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "claude" | "anthropic" => Some(ProviderId::Claude),
            "gemini" | "google" => Some(ProviderId::Gemini),
            _ => None,
        }
    }

    pub const ALL: [ProviderId; 2] = [ProviderId::Claude, ProviderId::Gemini];
}

// =============================================================================
// Neutral conversation model
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    /// The assistant. Named after Gemini's wire value because "model" is the
    /// less ambiguous word once more than one vendor is in play.
    Model,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    Text(String),
    Image {
        mime: String,
        data_base64: String,
    },
    ToolCall {
        id: String,
        name: String,
        args: Value,
    },
    ToolResult {
        id: String,
        name: String,
        content: Vec<Part>,
        is_error: bool,
    },
    /// Vendor-internal reasoning, replayed verbatim to the provider that
    /// produced it and dropped for any other.
    Reasoning {
        provider: ProviderId,
        raw: Value,
    },
}

impl Part {
    pub fn text(t: impl Into<String>) -> Self {
        Part::Text(t.into())
    }

    pub fn png(data_base64: impl Into<String>) -> Self {
        Part::Image {
            mime: "image/png".to_string(),
            data_base64: data_base64.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub role: Role,
    pub parts: Vec<Part>,
}

impl Turn {
    pub fn user(parts: Vec<Part>) -> Self {
        Self {
            role: Role::User,
            parts,
        }
    }

    pub fn model(parts: Vec<Part>) -> Self {
        Self {
            role: Role::Model,
            parts,
        }
    }

    pub fn user_text(t: impl Into<String>) -> Self {
        Self::user(vec![Part::text(t)])
    }
}

/// Drop reasoning parts belonging to a different provider.
///
/// Called when the active provider changes mid-conversation. Without this we
/// would send Anthropic thinking blocks to Gemini (or the reverse), which is at
/// best ignored and at worst a 400.
pub fn sanitize_history(history: &mut Vec<Turn>, active: ProviderId) {
    for turn in history.iter_mut() {
        turn.parts
            .retain(|p| !matches!(p, Part::Reasoning { provider, .. } if *provider != active));
    }
    // A turn that held nothing but foreign reasoning would now be empty, and
    // both APIs reject empty content.
    history.retain(|t| !t.parts.is_empty());
}

/// A tool offered to the model, described in plain JSON Schema.
///
/// Anthropic accepts this as-is; the Gemini provider translates it into the
/// OpenAPI subset that `functionDeclarations` requires.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Everything a provider needs to issue one request.
pub struct TurnRequest<'a> {
    pub model: &'a str,
    pub system: &'a str,
    pub effort: &'a str,
    pub max_tokens: u32,
    pub history: &'a [Turn],
    pub tools: &'a [ToolSpec],
}

/// Why the model stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    /// The model wants one or more tools executed.
    ToolUse,
    /// Declined on policy grounds. Anthropic reports `refusal`; Gemini reports
    /// `SAFETY` / `PROHIBITED_CONTENT` / `BLOCKLIST` / `SPII`.
    Refused {
        detail: Option<String>,
    },
    /// A server-side pause that is resumed by re-sending (Anthropic only).
    Paused,
    /// Anything else, kept as the raw vendor string for diagnosis.
    Other(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_input_tokens: u32,
}

/// One completed assistant turn, in neutral form.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssistantTurn {
    pub parts: Vec<Part>,
    pub stop: Option<StopReason>,
    pub usage: Usage,
    pub model: String,
}

impl AssistantTurn {
    pub fn text(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                Part::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Every tool call, in order. Each needs a matching result in the
    /// follow-up user turn.
    pub fn tool_calls(&self) -> Vec<(&str, &str, &Value)> {
        self.parts
            .iter()
            .filter_map(|p| match p {
                Part::ToolCall { id, name, args } => Some((id.as_str(), name.as_str(), args)),
                _ => None,
            })
            .collect()
    }
}

/// Something worth showing the user while a turn streams in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnEvent {
    TextDelta(String),
    ReasoningStarted,
    ToolUseStarted { name: String },
    Error(String),
    Done,
}

/// A failure while talking to a provider.
#[derive(Debug, Clone)]
pub enum ApiError {
    Network(String),
    /// Credentials were rejected. The caller invalidates the cached credential
    /// and retries once before surfacing this.
    Unauthorized(String),
    Http {
        status: u16,
        body: String,
    },
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Network(m) => write!(f, "network error: {m}"),
            ApiError::Unauthorized(m) => write!(f, "authentication rejected: {m}"),
            ApiError::Http { status, body } => write!(f, "API error {status}: {body}"),
        }
    }
}

/// One vendor's implementation.
///
/// Blocking on purpose: the worker owns the Tokio runtime and passes it in, so
/// implementations can `block_on` without the trait becoming async (and without
/// an `async_trait` dependency).
pub trait Provider: Send {
    fn id(&self) -> ProviderId;

    fn auth(&self) -> &dyn AuthBroker;

    fn send_turn(
        &mut self,
        runtime: &tokio::runtime::Runtime,
        request: &TurnRequest<'_>,
        on_event: &mut dyn FnMut(TurnEvent),
    ) -> Result<AssistantTurn, ApiError>;
}

/// Construct the provider for `id`.
pub fn build(id: ProviderId) -> Result<Box<dyn Provider>, ApiError> {
    match id {
        ProviderId::Claude => Ok(Box::new(anthropic::AnthropicProvider::new()?)),
        // Gemini lands in the next commit; until then, selecting it is a clear
        // error rather than a silent fallback to the other vendor.
        ProviderId::Gemini => Err(ApiError::Network(
            "the Gemini provider is not implemented yet; use `set ai_provider, claude`".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn provider_ids_round_trip_through_their_setting_value() {
        for id in ProviderId::ALL {
            assert_eq!(ProviderId::parse(id.as_str()), Some(id));
        }
    }

    #[test]
    fn provider_parsing_accepts_vendor_aliases_and_is_case_insensitive() {
        assert_eq!(ProviderId::parse("Anthropic"), Some(ProviderId::Claude));
        assert_eq!(ProviderId::parse("  GOOGLE "), Some(ProviderId::Gemini));
        assert_eq!(ProviderId::parse("gpt"), None);
    }

    #[test]
    fn each_provider_defaults_to_its_own_model() {
        assert!(ProviderId::Claude.default_model().starts_with("claude"));
        assert!(ProviderId::Gemini.default_model().starts_with("gemini"));
    }

    fn history_with_reasoning() -> Vec<Turn> {
        vec![
            Turn::user_text("hi"),
            Turn::model(vec![
                Part::Reasoning {
                    provider: ProviderId::Claude,
                    raw: json!({"signature": "sig"}),
                },
                Part::text("hello"),
            ]),
        ]
    }

    #[test]
    fn switching_provider_drops_foreign_reasoning_but_keeps_the_prose() {
        let mut h = history_with_reasoning();
        sanitize_history(&mut h, ProviderId::Gemini);
        assert_eq!(h[1].parts, vec![Part::text("hello")]);
    }

    #[test]
    fn staying_on_the_same_provider_preserves_reasoning_verbatim() {
        let mut h = history_with_reasoning();
        let before = h.clone();
        sanitize_history(&mut h, ProviderId::Claude);
        assert_eq!(h, before);
    }

    #[test]
    fn turns_left_empty_by_sanitizing_are_removed() {
        // Both APIs reject a turn with no content.
        let mut h = vec![
            Turn::user_text("hi"),
            Turn::model(vec![Part::Reasoning {
                provider: ProviderId::Claude,
                raw: json!({}),
            }]),
        ];
        sanitize_history(&mut h, ProviderId::Gemini);
        assert_eq!(h.len(), 1);
        assert!(h.iter().all(|t| !t.parts.is_empty()));
    }

    #[test]
    fn assistant_turn_extracts_text_and_tool_calls() {
        let turn = AssistantTurn {
            parts: vec![
                Part::text("Loading "),
                Part::text("it."),
                Part::ToolCall {
                    id: "c1".into(),
                    name: "run_command".into(),
                    args: json!({"commands": "load 1crn"}),
                },
            ],
            ..Default::default()
        };
        assert_eq!(turn.text(), "Loading it.");
        let calls = turn.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!((calls[0].0, calls[0].1), ("c1", "run_command"));
    }

    #[test]
    fn selecting_an_unimplemented_provider_errors_rather_than_silently_switching() {
        assert!(build(ProviderId::Gemini).is_err());
    }
}
