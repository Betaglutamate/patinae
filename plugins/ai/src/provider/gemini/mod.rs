//! The Gemini provider.
//!
//! Translates the neutral conversation model to and from the Gemini Developer
//! API. The wire types live in [`types`], the HTTP/SSE plumbing in [`http`], and
//! the chunk accumulator in [`accum`]; this module is the adapter.
//!
//! Three things differ enough from Anthropic to be worth stating up front:
//!
//! - **Gemini issues no tool-call ids.** A `functionCall` carries only a name,
//!   and the matching `functionResponse` is paired by name and position. Ids are
//!   therefore synthesized here so the vendor-neutral layer above can keep
//!   addressing calls by id.
//! - **Tool results cannot carry images.** A `functionResponse` is a JSON object,
//!   so a screenshot is re-anchored as an `inlineData` part in the same user
//!   turn, immediately after the response it belongs to.
//! - **Thought signatures attach to tool calls, which have nowhere to hold them**
//!   in the neutral model. They are kept in a side table on the provider, whose
//!   lifetime is exactly right: it is rebuilt whenever the provider or model
//!   changes, which is precisely when the signatures stop being valid.

pub mod accum;
pub mod http;
pub mod types;

use std::collections::HashMap;

use serde_json::{json, Value};

use super::{
    ApiError, AssistantTurn, ModelInfo, Origin, Part, Provider, ProviderId, Role, StopReason,
    ToolSpec, Turn, TurnEvent, TurnRequest, Usage,
};
use crate::auth::{google::GoogleBroker, AuthBroker};

/// Model used when `gemini_model` is unset.
pub const DEFAULT_MODEL: &str = "gemini-2.5-pro";

/// Offline fallback catalogue; `GET /v1beta/models` is the normal source.
pub fn builtin_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo::new(DEFAULT_MODEL, "Gemini 2.5 Pro").with_context(1_048_576),
        ModelInfo::new("gemini-2.5-flash", "Gemini 2.5 Flash").with_context(1_048_576),
    ]
}

pub struct GeminiProvider {
    client: reqwest::Client,
    broker: GoogleBroker,
    /// Thought signatures issued alongside tool calls, keyed by the synthesized
    /// call id.
    ///
    /// Gemini expects these echoed back on the `functionCall` when the
    /// conversation continues. The neutral [`Part::ToolCall`] has nowhere to put
    /// one, so they live here — and because the whole provider is rebuilt when
    /// the model changes, they cannot outlive the model that signed them.
    call_signatures: HashMap<String, String>,
}

impl GeminiProvider {
    pub fn new() -> Result<Self, ApiError> {
        Ok(Self {
            client: crate::provider::anthropic::http::build_client()?,
            broker: GoogleBroker,
            call_signatures: HashMap::new(),
        })
    }
}

impl Provider for GeminiProvider {
    fn id(&self) -> ProviderId {
        ProviderId::Gemini
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
        let credential = self
            .broker
            .credential()
            .map_err(|e| ApiError::Unauthorized(e.to_string()))?;

        let wire = types::GenerateContentRequest {
            // A turn whose every part was dropped — foreign reasoning that
            // `sanitize_history` did not reach, say — would serialize as a
            // `Content` with no parts, which the API rejects outright.
            contents: request
                .history
                .iter()
                .map(|t| to_wire_content(t, &self.call_signatures))
                .filter(|c| !c.parts.is_empty())
                .collect(),
            system_instruction: (!request.system.is_empty()).then(|| types::Content {
                role: None,
                parts: vec![types::Part::text(request.system)],
            }),
            tools: to_wire_tools(request.tools),
            generation_config: types::GenerationConfig {
                max_output_tokens: request.max_tokens,
                thinking_config: Some(thinking_for(request.effort)),
            },
        };

        let client = self.client.clone();
        let model = request.model.to_string();
        let turn = runtime.block_on(async move {
            http::stream_generate(&client, &credential, &model, &wire, |e| {
                on_event(to_neutral_event(e))
            })
            .await
        })?;

        let (neutral, signatures) = to_neutral_turn(turn, request.model);
        self.call_signatures.extend(signatures);
        Ok(neutral)
    }

    fn models(&self, runtime: &tokio::runtime::Runtime) -> Result<Vec<ModelInfo>, ApiError> {
        let Ok(credential) = self.broker.credential() else {
            return Ok(builtin_models());
        };
        let client = self.client.clone();
        match runtime.block_on(async move { http::list_models(&client, &credential).await }) {
            Ok(models) if !models.is_empty() => Ok(models),
            _ => Ok(builtin_models()),
        }
    }
}

// =============================================================================
// Reasoning effort
// =============================================================================

/// Translate the plugin's effort levels into a thinking budget.
///
/// Gemini takes a token allowance rather than a level, so the scale is mapped
/// onto one. `max` becomes `-1`, which asks the model to size its own budget —
/// a fixed ceiling would otherwise cap the highest setting *below* what the
/// model would choose. `includeThoughts` is always on: without it no
/// `thoughtSignature` is issued, and the model loses its reasoning across tool
/// round trips.
pub fn thinking_for(effort: &str) -> types::ThinkingConfig {
    let budget = match effort.trim().to_ascii_lowercase().as_str() {
        "low" => 4_096,
        "high" => 16_384,
        "xhigh" => 24_576,
        "max" => -1,
        // Anything unrecognised lands on the plugin's own default rather than
        // erroring: a typo in `patinaerc` should not make the agent unusable.
        _ => 8_192,
    };
    types::ThinkingConfig {
        include_thoughts: true,
        thinking_budget: budget,
    }
}

// =============================================================================
// Tool schemas
// =============================================================================

/// Keys Gemini's OpenAPI subset understands. Anything else is a 400.
const ALLOWED_SCHEMA_KEYS: &[&str] = &[
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
    "format",
    "nullable",
];

/// Rewrite a JSON Schema into the OpenAPI 3.0 subset `functionDeclarations`
/// accepts.
///
/// Two transformations. Unknown keys are dropped rather than passed through —
/// `additionalProperties` and `$schema` are rejected outright, and a tool schema
/// that grows one later should degrade rather than break the whole tool list.
/// And `type` is upper-cased, because the field is a proto enum whose JSON
/// mapping is case-sensitive.
pub fn to_wire_schema(schema: &Value) -> Value {
    let Some(object) = schema.as_object() else {
        return schema.clone();
    };

    let mut out = serde_json::Map::new();
    for (key, value) in object {
        if !ALLOWED_SCHEMA_KEYS.contains(&key.as_str()) {
            continue;
        }
        let translated = match key.as_str() {
            "type" => match value.as_str() {
                Some(t) => Value::String(t.to_ascii_uppercase()),
                None => value.clone(),
            },
            "properties" => match value.as_object() {
                Some(props) => Value::Object(
                    props
                        .iter()
                        .map(|(name, sub)| (name.clone(), to_wire_schema(sub)))
                        .collect(),
                ),
                None => value.clone(),
            },
            "items" => to_wire_schema(value),
            _ => value.clone(),
        };
        out.insert(key.clone(), translated);
    }
    Value::Object(out)
}

/// Whether a translated schema describes an object with no properties.
///
/// Such a declaration is rejected by the API, so `parameters` is dropped
/// entirely for argument-less tools like `get_scene_state`.
fn is_empty_object_schema(schema: &Value) -> bool {
    schema
        .get("properties")
        .and_then(|p| p.as_object())
        .map(|p| p.is_empty())
        .unwrap_or(true)
}

pub fn to_wire_tools(tools: &[ToolSpec]) -> Vec<types::Tool> {
    if tools.is_empty() {
        return Vec::new();
    }
    // Every function goes in a single `Tool`; the API treats a list of tools as
    // a list of *kinds* of tool, not a list of functions.
    vec![types::Tool {
        function_declarations: tools
            .iter()
            .map(|t| {
                let schema = to_wire_schema(&t.input_schema);
                types::FunctionDeclaration {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: (!is_empty_object_schema(&schema)).then_some(schema),
                }
            })
            .collect(),
    }]
}

// =============================================================================
// Neutral -> wire
// =============================================================================

fn to_wire_content(turn: &Turn, signatures: &HashMap<String, String>) -> types::Content {
    let mut parts = Vec::new();
    for part in &turn.parts {
        to_wire_parts(part, signatures, &mut parts);
    }
    types::Content::new(
        match turn.role {
            Role::User => types::Role::User,
            Role::Model => types::Role::Model,
        },
        parts,
    )
}

/// Append the wire form of one neutral part.
///
/// Appends rather than returns because a single tool result carrying a
/// screenshot becomes two parts: the `functionResponse` itself and the image
/// re-anchored beside it.
fn to_wire_parts(part: &Part, signatures: &HashMap<String, String>, out: &mut Vec<types::Part>) {
    match part {
        Part::Text(t) => out.push(types::Part::text(t)),
        Part::Image { mime, data_base64 } => out.push(types::Part::InlineData {
            inline_data: types::Blob {
                mime_type: mime.clone(),
                data: data_base64.clone(),
            },
        }),
        Part::ToolCall { id, name, args } => out.push(types::Part::FunctionCall {
            function_call: types::FunctionCall {
                name: name.clone(),
                args: args.clone(),
            },
            thought_signature: signatures.get(id).cloned(),
        }),
        Part::ToolResult {
            name,
            content,
            is_error,
            ..
        } => {
            // `response` must be a JSON object; a bare string is rejected.
            let text: String = content
                .iter()
                .filter_map(|p| match p {
                    Part::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            let key = if *is_error { "error" } else { "result" };
            out.push(types::Part::FunctionResponse {
                function_response: types::FunctionResponse {
                    name: name.clone(),
                    response: json!({ key: text }),
                },
            });

            // A functionResponse cannot hold an image, so any capture rides
            // alongside it in the same user turn instead of being dropped.
            for p in content {
                if let Part::Image { mime, data_base64 } = p {
                    out.push(types::Part::InlineData {
                        inline_data: types::Blob {
                            mime_type: mime.clone(),
                            data: data_base64.clone(),
                        },
                    });
                }
            }
        }
        // Replayed verbatim. `sanitize_history` has already discarded anything
        // from another model; this is the last-resort guard.
        Part::Reasoning { origin, raw } => {
            if origin.provider != ProviderId::Gemini {
                return;
            }
            if let Ok(p) = serde_json::from_value::<types::Part>(raw.clone()) {
                out.push(p);
            }
        }
    }
}

// =============================================================================
// Wire -> neutral
// =============================================================================

/// Convert an accumulated turn, returning the tool-call signatures to remember.
fn to_neutral_turn(
    turn: types::AssistantTurn,
    requested_model: &str,
) -> (AssistantTurn, HashMap<String, String>) {
    let origin = Origin::new(ProviderId::Gemini, requested_model);
    let mut parts = Vec::new();
    let mut signatures = HashMap::new();
    let mut call_ordinal = 0usize;

    for part in &turn.parts {
        match part {
            types::Part::Text {
                text,
                thought: Some(true),
                ..
            } => {
                let _ = text;
                parts.push(Part::Reasoning {
                    origin: origin.clone(),
                    raw: serde_json::to_value(part).unwrap_or(Value::Null),
                });
            }
            types::Part::Text { text, .. } => parts.push(Part::Text(text.clone())),
            types::Part::FunctionCall {
                function_call,
                thought_signature,
            } => {
                // Gemini supplies no id, so one is minted. The ordinal keeps it
                // unique when the same tool is called twice in one turn.
                let id = format!("{}-{}", function_call.name, call_ordinal);
                call_ordinal += 1;
                if let Some(sig) = thought_signature {
                    signatures.insert(id.clone(), sig.clone());
                }
                parts.push(Part::ToolCall {
                    id,
                    name: function_call.name.clone(),
                    args: function_call.args.clone(),
                });
            }
            types::Part::InlineData { inline_data } => parts.push(Part::Image {
                mime: inline_data.mime_type.clone(),
                data_base64: inline_data.data.clone(),
            }),
            // The model does not emit these; keep the turn lossless anyway.
            types::Part::FunctionResponse { .. } => parts.push(Part::Reasoning {
                origin: origin.clone(),
                raw: serde_json::to_value(part).unwrap_or(Value::Null),
            }),
        }
    }

    let has_tool_calls = parts.iter().any(|p| matches!(p, Part::ToolCall { .. }));
    let neutral = AssistantTurn {
        parts,
        stop: to_neutral_stop(&turn, has_tool_calls),
        usage: Usage {
            input_tokens: turn.usage.prompt_tokens,
            // Thinking tokens are billed as output but reported separately, so
            // a turn that thought hard would otherwise look free.
            output_tokens: turn.usage.candidates_tokens + turn.usage.thoughts_tokens,
            cached_input_tokens: turn.usage.cached_tokens,
        },
        model: turn.model,
    };
    (neutral, signatures)
}

fn to_neutral_stop(turn: &types::AssistantTurn, has_tool_calls: bool) -> Option<StopReason> {
    // A blocked *prompt* returns no candidate and therefore no finish reason,
    // but it is still a refusal and must not read as a normal end of turn.
    if let Some(reason) = &turn.block_reason {
        return Some(StopReason::Refused {
            detail: Some(format!("prompt blocked: {reason}")),
        });
    }

    let finish = turn.finish_reason?;
    Some(match finish {
        // Gemini reports STOP for a turn that ends in a tool call; the neutral
        // layer needs them distinguished or the agent loop stops one step early.
        types::FinishReason::Stop => {
            if has_tool_calls {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }
        }
        types::FinishReason::MaxTokens => StopReason::MaxTokens,
        f if f.is_refusal() => StopReason::Refused {
            detail: Some(format!("{f:?}")),
        },
        other => StopReason::Other(format!("{other:?}")),
    })
}

fn to_neutral_event(event: accum::TurnEvent) -> TurnEvent {
    match event {
        accum::TurnEvent::TextDelta(t) => TurnEvent::TextDelta(t),
        accum::TurnEvent::ThinkingStarted => TurnEvent::ReasoningStarted,
        accum::TurnEvent::ToolUseStarted { name } => TurnEvent::ToolUseStarted { name },
        accum::TurnEvent::Error(e) => TurnEvent::Error(e),
        accum::TurnEvent::Done => TurnEvent::Done,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn origin() -> Origin {
        Origin::new(ProviderId::Gemini, DEFAULT_MODEL)
    }

    fn wire_parts(part: &Part) -> Vec<types::Part> {
        let mut out = Vec::new();
        to_wire_parts(part, &HashMap::new(), &mut out);
        out
    }

    // --- schema translation ------------------------------------------------

    #[test]
    fn schema_types_are_upper_cased_at_every_level() {
        // The field is a proto enum; its JSON mapping is case-sensitive.
        let schema = to_wire_schema(&json!({
            "type": "object",
            "properties": {
                "commands": {"type": "string", "description": "d"},
                "sizes": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["commands"],
        }));

        assert_eq!(schema["type"], "OBJECT");
        assert_eq!(schema["properties"]["commands"]["type"], "STRING");
        assert_eq!(schema["properties"]["sizes"]["type"], "ARRAY");
        assert_eq!(schema["properties"]["sizes"]["items"]["type"], "INTEGER");
        assert_eq!(schema["required"][0], "commands");
        assert_eq!(schema["properties"]["commands"]["description"], "d");
    }

    #[test]
    fn keys_outside_the_openapi_subset_are_dropped() {
        // Passing these through is a 400 on the whole request, which would take
        // every other tool down with it.
        let schema = to_wire_schema(&json!({
            "type": "object",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "additionalProperties": false,
            "properties": {"a": {"type": "string", "default": "x", "examples": ["y"]}},
        }));

        assert!(schema.get("$schema").is_none());
        assert!(schema.get("additionalProperties").is_none());
        assert!(schema["properties"]["a"].get("default").is_none());
        assert!(schema["properties"]["a"].get("examples").is_none());
        assert_eq!(schema["properties"]["a"]["type"], "STRING");
    }

    #[test]
    fn an_argument_less_tool_sends_no_parameters_at_all() {
        // `{"type": "object", "properties": {}}` is rejected by the API.
        let tools = to_wire_tools(&[ToolSpec {
            name: "get_scene_state".into(),
            description: "d".into(),
            input_schema: json!({"type": "object", "properties": {}}),
        }]);
        assert_eq!(tools[0].function_declarations[0].parameters, None);
    }

    #[test]
    fn every_function_lands_in_one_tool_entry() {
        // A list of `Tool`s is a list of tool *kinds*, not of functions.
        let specs: Vec<ToolSpec> = ["a", "b", "c"]
            .iter()
            .map(|n| ToolSpec {
                name: (*n).into(),
                description: "d".into(),
                input_schema: json!({"type": "object", "properties": {"x": {"type": "string"}}}),
            })
            .collect();
        let tools = to_wire_tools(&specs);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function_declarations.len(), 3);
    }

    #[test]
    fn the_real_tool_schemas_all_translate() {
        for tool in crate::tools::schemas(true) {
            let schema = to_wire_schema(&tool.input_schema);
            assert_eq!(schema["type"], "OBJECT", "{}", tool.name);
            for key in schema.as_object().unwrap().keys() {
                assert!(
                    ALLOWED_SCHEMA_KEYS.contains(&key.as_str()),
                    "{} leaked `{key}` into the wire schema",
                    tool.name
                );
            }
        }
    }

    // --- effort ------------------------------------------------------------

    #[test]
    fn effort_maps_onto_a_thinking_budget_and_always_includes_thoughts() {
        assert_eq!(thinking_for("low").thinking_budget, 4_096);
        assert_eq!(thinking_for("medium").thinking_budget, 8_192);
        assert!(thinking_for("high").thinking_budget > thinking_for("medium").thinking_budget);
        // `max` asks the model to size its own budget rather than capping it.
        assert_eq!(thinking_for("max").thinking_budget, -1);
        // Without this no thoughtSignature is issued and reasoning is lost
        // across tool round trips.
        assert!(thinking_for("low").include_thoughts);
    }

    #[test]
    fn an_unrecognised_effort_falls_back_to_the_default_rather_than_failing() {
        assert_eq!(
            thinking_for("nonsense").thinking_budget,
            thinking_for("medium").thinking_budget
        );
        assert_eq!(thinking_for("HIGH").thinking_budget, 16_384);
    }

    // --- neutral -> wire ---------------------------------------------------

    #[test]
    fn tool_results_are_wrapped_in_an_object_because_a_bare_string_is_rejected() {
        let parts = wire_parts(&Part::ToolResult {
            id: "run_command-0".into(),
            name: "run_command".into(),
            content: vec![Part::text("done")],
            is_error: false,
        });
        match &parts[0] {
            types::Part::FunctionResponse { function_response } => {
                assert_eq!(function_response.name, "run_command");
                assert_eq!(function_response.response["result"], "done");
            }
            other => panic!("expected a function response, got {other:?}"),
        }
    }

    #[test]
    fn errored_tool_results_are_reported_under_an_error_key() {
        let parts = wire_parts(&Part::ToolResult {
            id: "c".into(),
            name: "run_python".into(),
            content: vec![Part::text("denied")],
            is_error: true,
        });
        match &parts[0] {
            types::Part::FunctionResponse { function_response } => {
                assert_eq!(function_response.response["error"], "denied");
            }
            other => panic!("expected a function response, got {other:?}"),
        }
    }

    #[test]
    fn a_screenshot_rides_beside_its_tool_result_rather_than_inside_it() {
        // functionResponse is a JSON object and cannot carry an image, so
        // dropping it here would silently blind the agent.
        let parts = wire_parts(&Part::ToolResult {
            id: "screenshot-0".into(),
            name: "screenshot".into(),
            content: vec![Part::text("capture:"), Part::png("QUJD")],
            is_error: false,
        });
        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], types::Part::FunctionResponse { .. }));
        match &parts[1] {
            types::Part::InlineData { inline_data } => {
                assert_eq!(inline_data.mime_type, "image/png");
                assert_eq!(inline_data.data, "QUJD");
            }
            other => panic!("expected inline image data, got {other:?}"),
        }
    }

    #[test]
    fn tool_calls_replay_the_signature_recorded_for_them() {
        let mut signatures = HashMap::new();
        signatures.insert("run_command-0".to_string(), "sig-abc".to_string());
        let mut out = Vec::new();
        to_wire_parts(
            &Part::ToolCall {
                id: "run_command-0".into(),
                name: "run_command".into(),
                args: json!({"commands": "load 1crn"}),
            },
            &signatures,
            &mut out,
        );
        match &out[0] {
            types::Part::FunctionCall {
                thought_signature, ..
            } => assert_eq!(thought_signature.as_deref(), Some("sig-abc")),
            other => panic!("expected a function call, got {other:?}"),
        }
    }

    #[test]
    fn a_tool_call_with_no_recorded_signature_omits_it() {
        let parts = wire_parts(&Part::ToolCall {
            id: "unknown".into(),
            name: "screenshot".into(),
            args: json!({}),
        });
        assert!(matches!(
            &parts[0],
            types::Part::FunctionCall {
                thought_signature: None,
                ..
            }
        ));
    }

    #[test]
    fn reasoning_from_another_provider_is_dropped_not_mangled() {
        let foreign = Part::Reasoning {
            origin: Origin::new(ProviderId::Claude, "claude-sonnet-5"),
            raw: json!({"type": "thinking", "thinking": "", "signature": "s"}),
        };
        assert!(wire_parts(&foreign).is_empty());
    }

    #[test]
    fn a_turn_left_with_no_parts_produces_no_content_at_all() {
        // Serializing it would send `{"parts": []}`, which the API rejects.
        let turn = Turn::model(vec![Part::Reasoning {
            origin: Origin::new(ProviderId::Claude, "claude-sonnet-5"),
            raw: json!({"type": "thinking"}),
        }]);
        let content = to_wire_content(&turn, &HashMap::new());
        assert!(content.parts.is_empty(), "the guard has nothing to catch");
        // …and the request builder is what drops it.
        let kept: Vec<_> = [content]
            .into_iter()
            .filter(|c| !c.parts.is_empty())
            .collect();
        assert!(kept.is_empty());
    }

    #[test]
    fn thought_parts_round_trip_byte_identically() {
        let wire = types::AssistantTurn {
            parts: vec![types::Part::Text {
                text: "reasoning".into(),
                thought: Some(true),
                thought_signature: Some("sig-xyz".into()),
            }],
            ..Default::default()
        };
        let (neutral, _) = to_neutral_turn(wire.clone(), DEFAULT_MODEL);
        let back = wire_parts(&neutral.parts[0]);
        assert_eq!(back, wire.parts);
    }

    // --- wire -> neutral ---------------------------------------------------

    #[test]
    fn synthesized_call_ids_stay_unique_across_repeated_tool_names() {
        // Two calls to the same tool in one turn must not collide, or the
        // second result would overwrite the first.
        let call = |name: &str| types::Part::FunctionCall {
            function_call: types::FunctionCall {
                name: name.into(),
                args: json!({}),
            },
            thought_signature: None,
        };
        let (neutral, _) = to_neutral_turn(
            types::AssistantTurn {
                parts: vec![call("run_command"), call("run_command")],
                ..Default::default()
            },
            DEFAULT_MODEL,
        );
        let ids: Vec<&str> = neutral.tool_calls().iter().map(|c| c.0).collect();
        assert_eq!(ids, ["run_command-0", "run_command-1"]);
    }

    #[test]
    fn signatures_are_keyed_by_the_id_that_was_minted_for_the_call() {
        let (_, signatures) = to_neutral_turn(
            types::AssistantTurn {
                parts: vec![types::Part::FunctionCall {
                    function_call: types::FunctionCall {
                        name: "screenshot".into(),
                        args: json!({}),
                    },
                    thought_signature: Some("sig".into()),
                }],
                ..Default::default()
            },
            DEFAULT_MODEL,
        );
        assert_eq!(
            signatures.get("screenshot-0").map(String::as_str),
            Some("sig")
        );
    }

    #[test]
    fn a_turn_ending_in_a_tool_call_reports_tool_use_not_end_turn() {
        // Gemini says STOP either way; reading it literally would end the agent
        // loop one step early and the tool would never run.
        let turn = types::AssistantTurn {
            parts: vec![types::Part::FunctionCall {
                function_call: types::FunctionCall {
                    name: "run_command".into(),
                    args: json!({}),
                },
                thought_signature: None,
            }],
            finish_reason: Some(types::FinishReason::Stop),
            ..Default::default()
        };
        let (neutral, _) = to_neutral_turn(turn, DEFAULT_MODEL);
        assert_eq!(neutral.stop, Some(StopReason::ToolUse));
    }

    #[test]
    fn a_plain_stop_is_an_end_of_turn() {
        let turn = types::AssistantTurn {
            parts: vec![types::Part::text("done")],
            finish_reason: Some(types::FinishReason::Stop),
            ..Default::default()
        };
        assert_eq!(
            to_neutral_turn(turn, DEFAULT_MODEL).0.stop,
            Some(StopReason::EndTurn)
        );
    }

    #[test]
    fn safety_finishes_become_refusals_carrying_their_reason() {
        for reason in [
            types::FinishReason::Safety,
            types::FinishReason::ProhibitedContent,
            types::FinishReason::Spii,
        ] {
            let turn = types::AssistantTurn {
                finish_reason: Some(reason),
                ..Default::default()
            };
            match to_neutral_turn(turn, DEFAULT_MODEL).0.stop {
                Some(StopReason::Refused { detail }) => {
                    assert!(detail.is_some(), "{reason:?} lost its explanation")
                }
                other => panic!("{reason:?} became {other:?}"),
            }
        }
    }

    #[test]
    fn a_blocked_prompt_is_a_refusal_even_with_no_finish_reason() {
        let turn = types::AssistantTurn {
            block_reason: Some("SAFETY".into()),
            ..Default::default()
        };
        match to_neutral_turn(turn, DEFAULT_MODEL).0.stop {
            Some(StopReason::Refused { detail }) => {
                assert!(detail.unwrap().contains("prompt blocked"))
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn thinking_tokens_are_counted_as_output() {
        // They are billed as output but reported separately, so a turn that
        // thought hard would otherwise look free.
        let turn = types::AssistantTurn {
            usage: types::UsageMetadata {
                prompt_tokens: 100,
                candidates_tokens: 20,
                cached_tokens: 90,
                thoughts_tokens: 500,
            },
            ..Default::default()
        };
        let (neutral, _) = to_neutral_turn(turn, DEFAULT_MODEL);
        assert_eq!(neutral.usage.output_tokens, 520);
        assert_eq!(neutral.usage.cached_input_tokens, 90);
    }

    #[test]
    fn reasoning_is_tagged_with_the_requested_model() {
        let turn = types::AssistantTurn {
            parts: vec![types::Part::Text {
                text: "t".into(),
                thought: Some(true),
                thought_signature: None,
            }],
            model: "gemini-2.5-pro-002".into(),
            ..Default::default()
        };
        let (neutral, _) = to_neutral_turn(turn, DEFAULT_MODEL);
        match &neutral.parts[0] {
            Part::Reasoning { origin: o, .. } => assert_eq!(*o, origin()),
            other => panic!("expected reasoning, got {other:?}"),
        }
    }
}
