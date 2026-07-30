//! The provider seam.
//!
//! Everything above this module — the agent loop, tools, panel, transcript,
//! prompt — is vendor-neutral. Everything below it translates to and from one
//! vendor's wire format. None of the three vendors ships a Rust SDK, so every
//! adapter is hand-written.
//!
//! The neutral conversation model is deliberately *not* any one vendor's shape.
//! Three details make it work across all of them:
//!
//! - **Tool calls carry both an `id` and a `name`.** Anthropic and OpenRouter
//!   pair results to calls by id; Gemini's `functionResponse` wants the name and
//!   supplies no id at all. Carrying both means no provider has to invent one.
//! - **Reasoning is opaque and tagged with its [`Origin`].** Anthropic requires
//!   thinking blocks echoed back byte-identically, Gemini has its own
//!   `thoughtSignature`, and OpenRouter forwards `reasoning_details` to whichever
//!   upstream vendor issued them. None is meaningful to any other, so
//!   [`sanitize_history`] drops foreign ones when the user switches. Every API
//!   tolerates their absence.
//! - **Capabilities are data, not assumptions.** [`ModelInfo`] carries what a
//!   model can actually do, because on OpenRouter the answer varies per model
//!   rather than per vendor.

pub mod anthropic;
pub mod gemini;
pub mod openrouter;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::auth::AuthBroker;

/// Which vendor backs the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderId {
    Claude,
    Gemini,
    OpenRouter,
}

impl ProviderId {
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderId::Claude => "claude",
            ProviderId::Gemini => "gemini",
            ProviderId::OpenRouter => "openrouter",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ProviderId::Claude => "Claude",
            ProviderId::Gemini => "Gemini",
            ProviderId::OpenRouter => "OpenRouter",
        }
    }

    /// Model used when the per-provider model setting is unset.
    ///
    /// Per-provider rather than one global default: the vendors' model names
    /// share no namespace, and users switch back and forth.
    ///
    /// OpenRouter defaults to a model that supports tools, images *and*
    /// reasoning, so every feature of the plugin works on the first run. Picking
    /// a cheaper model that cannot call tools would leave the agent inert with
    /// no obvious explanation.
    pub fn default_model(self) -> &'static str {
        match self {
            ProviderId::Claude => anthropic::types::DEFAULT_MODEL,
            ProviderId::Gemini => gemini::DEFAULT_MODEL,
            ProviderId::OpenRouter => openrouter::DEFAULT_MODEL,
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "claude" | "anthropic" => Some(ProviderId::Claude),
            "gemini" | "google" => Some(ProviderId::Gemini),
            "openrouter" | "open-router" | "or" => Some(ProviderId::OpenRouter),
            _ => None,
        }
    }

    /// Models to offer when the live catalogue cannot be reached.
    ///
    /// Every provider can list its models over HTTP, so this is a fallback for
    /// the offline / not-yet-signed-in case rather than the normal path. It is
    /// deliberately short: a stale hardcoded catalogue is worse than a small one,
    /// and the model setting accepts any string regardless.
    pub fn builtin_models(self) -> Vec<ModelInfo> {
        match self {
            ProviderId::Claude => anthropic::builtin_models(),
            ProviderId::Gemini => gemini::builtin_models(),
            ProviderId::OpenRouter => openrouter::builtin_models(),
        }
    }

    pub const ALL: [ProviderId; 3] = [
        ProviderId::Claude,
        ProviderId::Gemini,
        ProviderId::OpenRouter,
    ];
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

/// Which provider *and model* produced a piece of opaque reasoning.
///
/// Both halves matter. Anthropic validates a thinking block's signature against
/// the model that signed it, Gemini does the same with `thoughtSignature`, and
/// on OpenRouter a single provider spans every vendor at once — so provider
/// alone cannot decide whether a block is safe to replay. Recording the model
/// too means switching `claude_model` mid-conversation is handled by the same
/// rule that handles switching vendor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    pub provider: ProviderId,
    pub model: String,
}

impl Origin {
    pub fn new(provider: ProviderId, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
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
    /// Vendor-internal reasoning, replayed verbatim to the model that produced
    /// it and dropped for any other.
    Reasoning {
        origin: Origin,
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

/// Drop reasoning parts that did not come from `active`.
///
/// Called when the provider *or the model* changes mid-conversation. Without
/// this we would send Anthropic thinking blocks to Gemini (or a Sonnet
/// signature to Opus), which is at best ignored and at worst a 400.
pub fn sanitize_history(history: &mut Vec<Turn>, active: &Origin) {
    for turn in history.iter_mut() {
        turn.parts
            .retain(|p| !matches!(p, Part::Reasoning { origin, .. } if origin != active));
    }
    // A turn that held nothing but foreign reasoning would now be empty, and
    // every API rejects empty content.
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

// =============================================================================
// Model catalogue
// =============================================================================

/// Published price in US dollars per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Price {
    pub prompt: f64,
    pub completion: f64,
}

/// One model, and what it can actually do.
///
/// Capabilities are carried as data rather than inferred from the provider,
/// because on OpenRouter a single provider fronts hundreds of models whose
/// support for tools, images and reasoning all differ. The panel uses these to
/// filter the picker and to warn before a model is chosen that cannot drive the
/// viewer at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    /// Human-readable name for the picker; falls back to the id.
    pub label: String,
    /// Context window in tokens, or `None` when the vendor does not publish it.
    pub context_length: Option<u32>,
    /// Whether the model can call tools. **The agent is inert without this**, so
    /// it is the one capability the picker filters on rather than merely
    /// displays.
    pub tools: bool,
    /// Whether the model accepts image input, i.e. whether `screenshot` works.
    pub images: bool,
    /// Whether the model exposes a reasoning/thinking mode.
    pub reasoning: bool,
    pub price: Option<Price>,
}

impl ModelInfo {
    /// A model with no published capability metadata.
    ///
    /// Defaults are optimistic on purpose: this is used for the builtin
    /// fallback lists and for a model the user typed by hand, where refusing to
    /// offer a model we simply know nothing about is worse than offering it.
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            context_length: None,
            tools: true,
            images: true,
            reasoning: true,
            price: None,
        }
    }

    pub fn with_context(mut self, tokens: u32) -> Self {
        self.context_length = Some(tokens);
        self
    }

    /// A one-line capability summary for the panel.
    pub fn badges(&self) -> String {
        let mut parts = Vec::new();
        parts.push(if self.tools {
            "tools ✓"
        } else {
            "no tools ✗"
        });
        if self.images {
            parts.push("vision ✓");
        }
        if self.reasoning {
            parts.push("reasoning ✓");
        }
        let mut out = parts.join(" · ");
        if let Some(ctx) = self.context_length {
            out.push_str(&format!(" · {}k ctx", ctx / 1000));
        }
        if let Some(p) = self.price {
            out.push_str(&format!(
                " · ${:.2}/${:.2} per Mtok",
                p.prompt, p.completion
            ));
        }
        out
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

    /// The models this provider can serve, newest-relevant first.
    ///
    /// Every vendor publishes a models endpoint, so the default is only reached
    /// by a provider that has not overridden it. Implementations should fall
    /// back to [`ProviderId::builtin_models`] rather than erroring: an
    /// unreachable catalogue must degrade the picker, not break it.
    fn models(&self, _runtime: &tokio::runtime::Runtime) -> Result<Vec<ModelInfo>, ApiError> {
        Ok(self.id().builtin_models())
    }

    /// A status line better than the credential broker can produce alone.
    ///
    /// The broker only knows whether it found a credential. A provider can go
    /// further if the vendor exposes an endpoint describing it — OpenRouter
    /// reports the key's label and remaining credit, which is the question
    /// people actually have. Returning `None` keeps the broker's own answer.
    ///
    /// Only called when a runtime already exists, so a user who never signs in
    /// still pays nothing for it.
    fn status_line(&self, _runtime: &tokio::runtime::Runtime) -> Option<String> {
        None
    }
}

/// Construct the provider for `id`.
pub fn build(id: ProviderId) -> Result<Box<dyn Provider>, ApiError> {
    match id {
        ProviderId::Claude => Ok(Box::new(anthropic::AnthropicProvider::new()?)),
        ProviderId::Gemini => Ok(Box::new(gemini::GeminiProvider::new()?)),
        ProviderId::OpenRouter => Ok(Box::new(openrouter::OpenRouterProvider::new()?)),
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
        assert_eq!(
            ProviderId::parse(" OpenRouter "),
            Some(ProviderId::OpenRouter)
        );
        assert_eq!(ProviderId::parse("gpt"), None);
    }

    #[test]
    fn each_provider_defaults_to_its_own_model() {
        assert!(ProviderId::Claude.default_model().starts_with("claude"));
        assert!(ProviderId::Gemini.default_model().starts_with("gemini"));
        // OpenRouter ids are always `vendor/model`.
        assert!(ProviderId::OpenRouter.default_model().contains('/'));
    }

    #[test]
    fn every_provider_offers_a_non_empty_offline_fallback() {
        for id in ProviderId::ALL {
            let models = id.builtin_models();
            assert!(!models.is_empty(), "{} has no fallback", id.as_str());
            assert!(
                models.iter().any(|m| m.id == id.default_model()),
                "{}'s default model is missing from its fallback list",
                id.as_str()
            );
        }
    }

    fn claude_origin() -> Origin {
        Origin::new(ProviderId::Claude, "claude-sonnet-5")
    }

    fn history_with_reasoning() -> Vec<Turn> {
        vec![
            Turn::user_text("hi"),
            Turn::model(vec![
                Part::Reasoning {
                    origin: claude_origin(),
                    raw: json!({"signature": "sig"}),
                },
                Part::text("hello"),
            ]),
        ]
    }

    #[test]
    fn switching_provider_drops_foreign_reasoning_but_keeps_the_prose() {
        let mut h = history_with_reasoning();
        sanitize_history(&mut h, &Origin::new(ProviderId::Gemini, "gemini-2.5-pro"));
        assert_eq!(h[1].parts, vec![Part::text("hello")]);
    }

    #[test]
    fn switching_model_within_one_provider_also_drops_reasoning() {
        // A Sonnet thinking signature does not validate on Opus, so provider
        // identity alone is not enough to decide this.
        let mut h = history_with_reasoning();
        sanitize_history(&mut h, &Origin::new(ProviderId::Claude, "claude-opus-5"));
        assert_eq!(h[1].parts, vec![Part::text("hello")]);
    }

    #[test]
    fn staying_on_the_same_model_preserves_reasoning_verbatim() {
        let mut h = history_with_reasoning();
        let before = h.clone();
        sanitize_history(&mut h, &claude_origin());
        assert_eq!(h, before);
    }

    #[test]
    fn turns_left_empty_by_sanitizing_are_removed() {
        // Every API rejects a turn with no content.
        let mut h = vec![
            Turn::user_text("hi"),
            Turn::model(vec![Part::Reasoning {
                origin: claude_origin(),
                raw: json!({}),
            }]),
        ];
        sanitize_history(&mut h, &Origin::new(ProviderId::Gemini, "gemini-2.5-pro"));
        assert_eq!(h.len(), 1);
        assert!(h.iter().all(|t| !t.parts.is_empty()));
    }

    #[test]
    fn badges_lead_with_tool_support_and_flag_its_absence() {
        let mut m = ModelInfo::new("x/y", "Y").with_context(200_000);
        assert!(m.badges().starts_with("tools ✓"));
        assert!(m.badges().contains("200k ctx"));

        m.tools = false;
        assert!(
            m.badges().starts_with("no tools ✗"),
            "a model that cannot drive the viewer must say so first"
        );
    }

    #[test]
    fn badges_quote_prices_per_million_tokens() {
        let mut m = ModelInfo::new("x/y", "Y");
        m.price = Some(Price {
            prompt: 3.0,
            completion: 15.0,
        });
        assert!(m.badges().contains("$3.00/$15.00 per Mtok"));
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
    fn every_provider_id_builds_an_adapter_reporting_that_same_id() {
        // Guards against a `build` arm wired to the wrong adapter, which would
        // silently send one vendor's traffic to another's credentials.
        for id in ProviderId::ALL {
            let provider = build(id).expect("every provider is implemented");
            assert_eq!(provider.id(), id);
        }
    }
}
