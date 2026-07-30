//! The OpenRouter provider.
//!
//! One endpoint fronting hundreds of models from every vendor, speaking the
//! OpenAI Chat Completions dialect. The wire types live in [`types`], the
//! HTTP/SSE plumbing in [`http`], and the chunk accumulator in [`accum`]; this
//! module is the adapter.
//!
//! What makes it different from the other two providers:
//!
//! - **The conversation is a flat list of messages, not a list of turns.** One
//!   neutral [`Turn`] carrying several tool results becomes several messages, so
//!   the translation is a flattening rather than a mapping.
//! - **A `tool` message cannot carry an image.** Its content is a plain string,
//!   so a screenshot is re-anchored into a following `user` message — see
//!   [`push_user_turn`]. Dropping it instead would silently blind the agent
//!   every time it tried to look at its own work.
//! - **Capabilities vary per model rather than per provider.** A model without
//!   tool support cannot drive the viewer at all, which is why [`ModelInfo`]
//!   exists and why the panel filters on it.

pub mod accum;
pub mod http;
pub mod types;

use serde_json::Value;

use super::{
    ApiError, AssistantTurn, ModelInfo, Origin, Part, Provider, ProviderId, Role, StopReason,
    ToolSpec, Turn, TurnEvent, TurnRequest, Usage,
};
use crate::auth::{openrouter::OpenRouterBroker, AuthBroker, Credential};

/// Model used when `openrouter_model` is unset.
///
/// Chosen because it supports tools, images *and* reasoning, so every feature of
/// the plugin works on the first run. A cheaper default that could not call
/// tools would leave the agent inert with no obvious explanation.
pub const DEFAULT_MODEL: &str = "anthropic/claude-sonnet-5";

/// Offline fallback catalogue.
///
/// `GET /api/v1/models` is public and needs no key, so this is only reached with
/// no network at all. Kept to models this plugin already knows by name from the
/// two native providers, rather than a guessed cross-section of the catalogue.
pub fn builtin_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo::new(DEFAULT_MODEL, "Anthropic: Claude Sonnet 5").with_context(200_000),
        ModelInfo::new("anthropic/claude-opus-5", "Anthropic: Claude Opus 5").with_context(200_000),
        ModelInfo::new("google/gemini-2.5-pro", "Google: Gemini 2.5 Pro").with_context(1_048_576),
        ModelInfo::new("google/gemini-2.5-flash", "Google: Gemini 2.5 Flash")
            .with_context(1_048_576),
    ]
}

pub struct OpenRouterProvider {
    client: reqwest::Client,
    broker: OpenRouterBroker,
}

impl OpenRouterProvider {
    pub fn new() -> Result<Self, ApiError> {
        Ok(Self {
            client: crate::provider::anthropic::http::build_client()?,
            broker: OpenRouterBroker,
        })
    }

    fn key(&self) -> Result<String, ApiError> {
        match self.broker.credential() {
            Ok(Credential::Bearer(key)) => Ok(key),
            Ok(other) => Err(ApiError::Unauthorized(format!(
                "unexpected credential kind for OpenRouter: {other:?}"
            ))),
            Err(e) => Err(ApiError::Unauthorized(e.to_string())),
        }
    }
}

impl Provider for OpenRouterProvider {
    fn id(&self) -> ProviderId {
        ProviderId::OpenRouter
    }

    fn auth(&self) -> &dyn AuthBroker {
        &self.broker
    }

    fn send_turn(
        &mut self,
        runtime: &tokio::runtime::Runtime,
        request: &TurnRequest<'_>,
        on_event: &mut dyn FnMut(TurnEvent),
    ) -> Result<AssistantTurn, ApiError> {
        let key = self.key()?;

        let mut messages = Vec::new();
        if !request.system.is_empty() {
            messages.push(types::Message::text(types::Role::System, request.system));
        }
        for turn in request.history {
            push_turn(turn, &mut messages);
        }

        let wire = types::ChatRequest {
            model: request.model.to_string(),
            messages,
            tools: to_wire_tools(request.tools),
            reasoning: Some(types::Reasoning {
                effort: to_wire_effort(request.effort).to_string(),
            }),
            max_tokens: request.max_tokens,
            stream: true,
        };

        let client = self.client.clone();
        let turn = runtime.block_on(async move {
            http::stream_chat(&client, &key, &wire, |e| on_event(to_neutral_event(e))).await
        })?;

        Ok(to_neutral_turn(turn, request.model))
    }

    fn models(&self, runtime: &tokio::runtime::Runtime) -> Result<Vec<ModelInfo>, ApiError> {
        let client = self.client.clone();
        match runtime.block_on(async move { http::list_models(&client).await }) {
            Ok(models) if !models.is_empty() => Ok(models),
            _ => Ok(builtin_models()),
        }
    }

    /// Report the key's label and remaining credit.
    ///
    /// Falls back to the broker's plain "Signed in with …" when the call fails,
    /// by returning `None` — a status line is not worth surfacing an error for.
    fn status_line(&self, runtime: &tokio::runtime::Runtime) -> Option<String> {
        let key = self.key().ok()?;
        let client = self.client.clone();
        runtime
            .block_on(async move { http::describe_key(&client, &key).await })
            .ok()
    }
}

// =============================================================================
// Reasoning effort
// =============================================================================

/// Map the plugin's five effort levels onto the three OpenRouter defines.
///
/// `xhigh` and `max` clamp to `high` rather than erroring: they are meaningful
/// on Anthropic's own API, and a user who set one there should not have the
/// agent refuse to run when they switch provider.
pub fn to_wire_effort(effort: &str) -> &'static str {
    match effort.trim().to_ascii_lowercase().as_str() {
        "low" => "low",
        "high" | "xhigh" | "max" => "high",
        _ => "medium",
    }
}

// =============================================================================
// Neutral -> wire
// =============================================================================

pub fn to_wire_tools(tools: &[ToolSpec]) -> Vec<types::Tool> {
    tools
        .iter()
        .map(|t| types::Tool {
            kind: "function".to_string(),
            function: types::FunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                // The OpenAI dialect takes JSON Schema directly.
                parameters: t.input_schema.clone(),
            },
        })
        .collect()
}

/// Flatten one neutral turn into the messages it becomes.
fn push_turn(turn: &Turn, out: &mut Vec<types::Message>) {
    match turn.role {
        Role::User => push_user_turn(turn, out),
        Role::Model => push_model_turn(turn, out),
    }
}

/// A user turn becomes one `tool` message per tool result, then at most one
/// `user` message for everything else.
///
/// The split exists because tool results and ordinary user content cannot share
/// a message, and because images cannot live in a tool result at all. Any image
/// a tool returned is re-anchored into the trailing user message, which is the
/// only place the API will accept it.
fn push_user_turn(turn: &Turn, out: &mut Vec<types::Message>) {
    let mut carried: Vec<types::ContentPart> = Vec::new();

    for part in &turn.parts {
        match part {
            Part::ToolResult {
                id,
                content,
                is_error,
                ..
            } => {
                let mut text: String = content
                    .iter()
                    .filter_map(|p| match p {
                        Part::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let images: Vec<&Part> = content
                    .iter()
                    .filter(|p| matches!(p, Part::Image { .. }))
                    .collect();
                if !images.is_empty() {
                    // Say where the image went. Without this the model sees a
                    // tool result claiming to be a capture with no image in it,
                    // and reliably concludes the screenshot failed.
                    text.push_str(
                        "\n(The image is attached in the following user message, because a tool \
                         result cannot carry one.)",
                    );
                }
                // An empty string reads as a tool that returned nothing rather
                // than one that succeeded quietly.
                if text.trim().is_empty() {
                    text = "(no output)".to_string();
                }
                if *is_error {
                    text = format!("Error: {text}");
                }

                out.push(types::Message::tool_result(id, text));

                for image in images {
                    if let Part::Image { mime, data_base64 } = image {
                        carried.push(types::ContentPart::ImageUrl {
                            image_url: types::ImageUrl::data_uri(mime, data_base64),
                        });
                    }
                }
            }
            Part::Text(t) => carried.push(types::ContentPart::Text { text: t.clone() }),
            Part::Image { mime, data_base64 } => carried.push(types::ContentPart::ImageUrl {
                image_url: types::ImageUrl::data_uri(mime, data_base64),
            }),
            // A user turn carries no reasoning or tool calls.
            _ => {}
        }
    }

    if carried.is_empty() {
        return;
    }
    // A lone text part is sent as a bare string: it is the overwhelmingly common
    // case and the array form is accepted less widely across upstream vendors.
    let content = match carried.as_slice() {
        [types::ContentPart::Text { text }] => types::Content::Text(text.clone()),
        _ => types::Content::Parts(carried),
    };
    out.push(types::Message::new(types::Role::User, content));
}

fn push_model_turn(turn: &Turn, out: &mut Vec<types::Message>) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut reasoning_details = Vec::new();

    for part in &turn.parts {
        match part {
            Part::Text(t) => text.push_str(t),
            Part::ToolCall { id, name, args } => {
                // Arguments go back as the JSON *string* they arrived as.
                tool_calls.push(types::ToolCall::function(
                    id,
                    name,
                    serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string()),
                ));
            }
            Part::Reasoning { origin, raw } => {
                // `sanitize_history` has already discarded anything from another
                // model; this is the last-resort guard.
                if origin.provider != ProviderId::OpenRouter {
                    continue;
                }
                match raw {
                    Value::Array(details) => reasoning_details.extend(details.iter().cloned()),
                    other => reasoning_details.push(other.clone()),
                }
            }
            _ => {}
        }
    }

    if text.is_empty() && tool_calls.is_empty() {
        return;
    }
    out.push(types::Message {
        role: Some(types::Role::Assistant),
        content: (!text.is_empty()).then_some(types::Content::Text(text)),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id: None,
        reasoning_details: (!reasoning_details.is_empty()).then_some(reasoning_details),
    });
}

// =============================================================================
// Wire -> neutral
// =============================================================================

fn to_neutral_turn(turn: types::AssistantTurn, requested_model: &str) -> AssistantTurn {
    let mut parts = Vec::new();

    if !turn.reasoning_details.is_empty() {
        // Kept as one array so the sequence is replayed exactly as the model
        // produced it — upstream vendors reject a reordered trace.
        parts.push(Part::Reasoning {
            origin: Origin::new(ProviderId::OpenRouter, requested_model),
            raw: Value::Array(turn.reasoning_details),
        });
    }
    if !turn.text.is_empty() {
        parts.push(Part::Text(turn.text));
    }

    let has_calls = !turn.tool_calls.is_empty();
    for call in turn.tool_calls {
        parts.push(Part::ToolCall {
            id: call.id,
            name: call.function.name,
            // Models emit invalid JSON here often enough to be routine; an
            // empty object lets the tool report a useful error instead of the
            // whole turn failing.
            args: serde_json::from_str(call.function.arguments.trim())
                .unwrap_or_else(|_| serde_json::json!({})),
        });
    }

    AssistantTurn {
        parts,
        stop: Some(to_neutral_stop(turn.finish_reason.as_deref(), has_calls)),
        usage: Usage {
            input_tokens: turn.usage.prompt_tokens,
            output_tokens: turn.usage.completion_tokens,
            // OpenRouter does not break out cache reads in the streaming usage
            // block, so this is left at zero rather than guessed.
            cached_input_tokens: 0,
        },
        model: turn.model,
    }
}

fn to_neutral_stop(finish_reason: Option<&str>, has_calls: bool) -> StopReason {
    // Some upstream providers report `stop` even when the turn ends in a tool
    // call. Trusting the string alone would end the agent loop one step early
    // and the tool would never run.
    if has_calls {
        return StopReason::ToolUse;
    }
    match finish_reason {
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("content_filter") => StopReason::Refused {
            detail: Some("blocked by the provider's content filter".to_string()),
        },
        Some("stop") | None => StopReason::EndTurn,
        Some(other) => StopReason::Other(other.to_string()),
    }
}

fn to_neutral_event(event: accum::TurnEvent) -> TurnEvent {
    match event {
        accum::TurnEvent::TextDelta(t) => TurnEvent::TextDelta(t),
        accum::TurnEvent::ReasoningStarted => TurnEvent::ReasoningStarted,
        accum::TurnEvent::ToolUseStarted { name } => TurnEvent::ToolUseStarted { name },
        accum::TurnEvent::Error(e) => TurnEvent::Error(e),
        accum::TurnEvent::Done => TurnEvent::Done,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn messages(turn: &Turn) -> Vec<types::Message> {
        let mut out = Vec::new();
        push_turn(turn, &mut out);
        out
    }

    fn text_of(message: &types::Message) -> String {
        match &message.content {
            Some(types::Content::Text(t)) => t.clone(),
            Some(types::Content::Parts(parts)) => parts
                .iter()
                .filter_map(|p| match p {
                    types::ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect(),
            None => String::new(),
        }
    }

    // --- effort ------------------------------------------------------------

    #[test]
    fn effort_clamps_to_the_three_levels_openrouter_defines() {
        assert_eq!(to_wire_effort("low"), "low");
        assert_eq!(to_wire_effort("medium"), "medium");
        assert_eq!(to_wire_effort("high"), "high");
        // Meaningful on Anthropic's own API; must not break a provider switch.
        assert_eq!(to_wire_effort("xhigh"), "high");
        assert_eq!(to_wire_effort("max"), "high");
        assert_eq!(to_wire_effort("nonsense"), "medium");
    }

    // --- user turns --------------------------------------------------------

    #[test]
    fn a_plain_prompt_becomes_one_user_message_with_string_content() {
        let out = messages(&Turn::user_text("hello"));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, Some(types::Role::User));
        assert_eq!(out[0].content, Some(types::Content::Text("hello".into())));
    }

    #[test]
    fn each_tool_result_becomes_its_own_tool_message_keyed_by_call_id() {
        let out = messages(&Turn::user(vec![
            Part::ToolResult {
                id: "call_1".into(),
                name: "run_command".into(),
                content: vec![Part::text("done")],
                is_error: false,
            },
            Part::ToolResult {
                id: "call_2".into(),
                name: "count_atoms".into(),
                content: vec![Part::text("42 atoms")],
                is_error: false,
            },
        ]));

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(out[1].tool_call_id.as_deref(), Some("call_2"));
        assert_eq!(text_of(&out[1]), "42 atoms");
    }

    #[test]
    fn a_screenshot_is_re_anchored_into_a_following_user_message() {
        // A tool message's content is a plain string, so the image cannot ride
        // inside it. Dropping it would silently blind the agent.
        let out = messages(&Turn::user(vec![Part::ToolResult {
            id: "call_1".into(),
            name: "screenshot".into(),
            content: vec![Part::text("Viewport capture, 1024x768:"), Part::png("QUJD")],
            is_error: false,
        }]));

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, Some(types::Role::Tool));
        assert_eq!(out[1].role, Some(types::Role::User));
        match out[1].content.as_ref().unwrap() {
            types::Content::Parts(parts) => match &parts[0] {
                types::ContentPart::ImageUrl { image_url } => {
                    assert_eq!(image_url.url, "data:image/png;base64,QUJD")
                }
                other => panic!("expected an image, got {other:?}"),
            },
            other => panic!("expected content parts, got {other:?}"),
        }
    }

    #[test]
    fn a_tool_result_holding_an_image_says_where_the_image_went() {
        // Otherwise the model sees a result claiming to be a capture with no
        // image in it, and concludes the screenshot failed.
        let out = messages(&Turn::user(vec![Part::ToolResult {
            id: "c".into(),
            name: "screenshot".into(),
            content: vec![Part::text("capture:"), Part::png("QUJD")],
            is_error: false,
        }]));
        assert!(text_of(&out[0]).contains("following user message"));
    }

    #[test]
    fn errored_tool_results_are_marked_because_the_wire_has_no_error_flag() {
        // Unlike Anthropic there is no `is_error` field here, so it has to be
        // carried in the text or the model reads a failure as a success.
        let out = messages(&Turn::user(vec![Part::ToolResult {
            id: "c".into(),
            name: "run_python".into(),
            content: vec![Part::text("denied by the user")],
            is_error: true,
        }]));
        assert!(text_of(&out[0]).starts_with("Error: "));
    }

    #[test]
    fn a_tool_that_returned_nothing_does_not_send_an_empty_string() {
        let out = messages(&Turn::user(vec![Part::ToolResult {
            id: "c".into(),
            name: "run_command".into(),
            content: vec![],
            is_error: false,
        }]));
        assert_eq!(text_of(&out[0]), "(no output)");
    }

    // --- model turns -------------------------------------------------------

    #[test]
    fn tool_calls_are_serialized_back_into_an_arguments_string() {
        let out = messages(&Turn::model(vec![Part::ToolCall {
            id: "call_1".into(),
            name: "run_command".into(),
            args: json!({"commands": "load 1crn"}),
        }]));

        let calls = out[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].kind, "function");
        assert_eq!(calls[0].function.arguments, r#"{"commands":"load 1crn"}"#);
        // An assistant message that is only tool calls has no content field.
        assert!(out[0].content.is_none());
    }

    #[test]
    fn reasoning_is_replayed_as_the_array_the_model_produced() {
        let details = json!([
            {"type": "reasoning.text", "index": 0, "text": "a", "signature": "s"},
            {"type": "reasoning.text", "index": 1, "text": "b"},
        ]);
        let out = messages(&Turn::model(vec![
            Part::Reasoning {
                origin: Origin::new(ProviderId::OpenRouter, DEFAULT_MODEL),
                raw: details.clone(),
            },
            Part::text("hi"),
        ]));
        assert_eq!(
            out[0].reasoning_details.as_ref().unwrap(),
            details.as_array().unwrap()
        );
    }

    #[test]
    fn reasoning_from_another_provider_is_dropped_not_mangled() {
        let out = messages(&Turn::model(vec![
            Part::Reasoning {
                origin: Origin::new(ProviderId::Claude, "claude-sonnet-5"),
                raw: json!({"type": "thinking", "signature": "s"}),
            },
            Part::text("hi"),
        ]));
        assert!(out[0].reasoning_details.is_none());
    }

    #[test]
    fn an_empty_model_turn_emits_no_message() {
        // Every API rejects a message with neither content nor tool calls.
        assert!(messages(&Turn::model(vec![])).is_empty());
    }

    // --- stop reasons ------------------------------------------------------

    #[test]
    fn a_turn_with_tool_calls_reports_tool_use_whatever_the_finish_reason_says() {
        // Some upstream providers report `stop` here; reading it literally ends
        // the agent loop one step early and the tool never runs.
        assert_eq!(to_neutral_stop(Some("stop"), true), StopReason::ToolUse);
        assert_eq!(
            to_neutral_stop(Some("tool_calls"), true),
            StopReason::ToolUse
        );
        assert_eq!(to_neutral_stop(None, true), StopReason::ToolUse);
    }

    #[test]
    fn ordinary_finish_reasons_map_across() {
        assert_eq!(to_neutral_stop(Some("stop"), false), StopReason::EndTurn);
        assert_eq!(
            to_neutral_stop(Some("length"), false),
            StopReason::MaxTokens
        );
        assert!(matches!(
            to_neutral_stop(Some("content_filter"), false),
            StopReason::Refused { .. }
        ));
        assert_eq!(
            to_neutral_stop(Some("something_new"), false),
            StopReason::Other("something_new".into())
        );
    }

    // --- round trip --------------------------------------------------------

    #[test]
    fn malformed_tool_arguments_degrade_to_an_empty_object() {
        let turn = types::AssistantTurn {
            tool_calls: vec![types::ToolCall::function("c", "screenshot", "{oops".into())],
            ..Default::default()
        };
        let neutral = to_neutral_turn(turn, DEFAULT_MODEL);
        assert_eq!(*neutral.tool_calls()[0].2, json!({}));
    }

    #[test]
    fn a_reasoning_array_survives_the_round_trip_intact() {
        let wire = types::AssistantTurn {
            text: "hi".into(),
            reasoning_details: vec![json!({"type": "reasoning.text", "text": "t"})],
            ..Default::default()
        };
        let neutral = to_neutral_turn(wire.clone(), DEFAULT_MODEL);
        let back = messages(&Turn::model(neutral.parts));
        assert_eq!(
            back[0].reasoning_details.as_ref().unwrap(),
            &wire.reasoning_details
        );
    }

    #[test]
    fn every_tool_call_in_a_turn_gets_exactly_one_tool_message_back() {
        // The API rejects a follow-up that leaves any call unanswered, so the
        // two halves have to stay in step.
        let assistant = Turn::model(vec![
            Part::ToolCall {
                id: "a".into(),
                name: "screenshot".into(),
                args: json!({}),
            },
            Part::ToolCall {
                id: "b".into(),
                name: "count_atoms".into(),
                args: json!({"selection": "all"}),
            },
        ]);
        let results = Turn::user(vec![
            Part::ToolResult {
                id: "a".into(),
                name: "screenshot".into(),
                content: vec![Part::text("ok"), Part::png("QUJD")],
                is_error: false,
            },
            Part::ToolResult {
                id: "b".into(),
                name: "count_atoms".into(),
                content: vec![Part::text("42")],
                is_error: false,
            },
        ]);

        let mut out = Vec::new();
        push_turn(&assistant, &mut out);
        push_turn(&results, &mut out);

        let called: Vec<&str> = out[0]
            .tool_calls
            .as_ref()
            .unwrap()
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        let answered: Vec<&str> = out
            .iter()
            .filter_map(|m| m.tool_call_id.as_deref())
            .collect();
        assert_eq!(called, answered);
        // …and the re-anchored image trails behind, not between them.
        assert_eq!(out.last().unwrap().role, Some(types::Role::User));
    }

    #[test]
    fn the_default_model_is_one_the_fallback_catalogue_describes_as_capable() {
        let model = builtin_models()
            .into_iter()
            .find(|m| m.id == DEFAULT_MODEL)
            .expect("the default model must be in the fallback list");
        assert!(model.tools && model.images && model.reasoning);
    }
}
