//! The Anthropic provider.
//!
//! Translates the neutral conversation model to and from the Messages API wire
//! format. The wire types live in [`types`], the HTTP/SSE plumbing in [`http`],
//! and the event accumulator in [`accum`]; this module is only the adapter.

pub mod accum;
pub mod http;
pub mod types;

use serde_json::Value;

use super::{
    ApiError, AssistantTurn, ModelInfo, Origin, Part, Provider, ProviderId, Role, StopReason,
    ToolSpec, Turn, TurnEvent, TurnRequest, Usage,
};
use crate::auth::{ant::AntBroker, AuthBroker, Credential};

/// Offline fallback catalogue.
///
/// `GET /v1/models` is the normal source; this covers the not-yet-signed-in
/// case. Anthropic publishes no capability metadata, so every current model is
/// described with the defaults — all of them call tools, read images and think.
pub fn builtin_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo::new(types::DEFAULT_MODEL, "Claude Sonnet 5"),
        ModelInfo::new("claude-opus-5", "Claude Opus 5"),
        ModelInfo::new("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ]
}

pub struct AnthropicProvider {
    client: reqwest::Client,
    broker: AntBroker,
}

impl AnthropicProvider {
    pub fn new() -> Result<Self, ApiError> {
        Ok(Self {
            client: http::build_client()?,
            broker: AntBroker,
        })
    }
}

impl Provider for AnthropicProvider {
    fn id(&self) -> ProviderId {
        ProviderId::Claude
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
        let token = match self.broker.credential() {
            Ok(Credential::Bearer(t)) => t,
            // The Anthropic broker only ever mints bearer tokens; anything else
            // is a programming error, not a user-facing auth failure.
            Ok(other) => {
                return Err(ApiError::Unauthorized(format!(
                    "unexpected credential kind for Anthropic: {other:?}"
                )))
            }
            Err(e) => return Err(ApiError::Unauthorized(e.to_string())),
        };

        let wire = types::MessagesRequest {
            model: request.model.to_string(),
            max_tokens: request.max_tokens,
            messages: request.history.iter().map(to_wire_message).collect(),
            system: if request.system.is_empty() {
                vec![]
            } else {
                vec![types::SystemBlock::cached(request.system)]
            },
            // Anthropic takes JSON Schema directly, so tools pass through.
            tools: request
                .tools
                .iter()
                .map(|t| types::Tool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.input_schema.clone(),
                })
                .collect(),
            output_config: Some(types::OutputConfig {
                effort: Some(request.effort.to_string()),
            }),
            stream: true,
        };

        let client = self.client.clone();
        let turn = runtime.block_on(async move {
            http::stream_message(&client, &token, &wire, |e| on_event(to_neutral_event(e))).await
        })?;

        // Tagged with the *requested* model, not the one the response reports.
        // Anthropic resolves an alias like `claude-sonnet-5` to a dated id in
        // its reply, and tagging with that would make every block look foreign
        // to the next turn's check and be discarded immediately.
        Ok(to_neutral_turn(turn, request.model))
    }

    fn models(&self, runtime: &tokio::runtime::Runtime) -> Result<Vec<ModelInfo>, ApiError> {
        let Ok(Credential::Bearer(token)) = self.broker.credential() else {
            return Ok(builtin_models());
        };
        let client = self.client.clone();
        match runtime.block_on(async move { http::list_models(&client, &token).await }) {
            Ok(models) if !models.is_empty() => Ok(models),
            // An unreachable catalogue must degrade the picker, not break it.
            _ => Ok(builtin_models()),
        }
    }
}

// =============================================================================
// Neutral -> wire
// =============================================================================

fn to_wire_message(turn: &Turn) -> types::Message {
    types::Message {
        role: match turn.role {
            Role::User => types::Role::User,
            Role::Model => types::Role::Assistant,
        },
        content: turn.parts.iter().filter_map(to_wire_block).collect(),
    }
}

fn to_wire_block(part: &Part) -> Option<types::ContentBlock> {
    Some(match part {
        Part::Text(t) => types::ContentBlock::Text { text: t.clone() },
        Part::Image { mime, data_base64 } => types::ContentBlock::Image {
            source: types::ImageSource::Base64 {
                media_type: mime.clone(),
                data: data_base64.clone(),
            },
        },
        Part::ToolCall { id, name, args } => types::ContentBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: args.clone(),
        },
        Part::ToolResult {
            id,
            content,
            is_error,
            ..
        } => types::ContentBlock::ToolResult {
            tool_use_id: id.clone(),
            content: content.iter().filter_map(to_wire_result_part).collect(),
            is_error: is_error.then_some(true),
        },
        // Replayed verbatim — the API rejects edited thinking blocks. A block
        // we can no longer parse is dropped rather than sent malformed.
        //
        // `sanitize_history` has already discarded anything from another model;
        // this is the last-resort guard against a foreign block reaching the
        // wire if that ever fails to run.
        Part::Reasoning { origin, raw } => {
            if origin.provider != ProviderId::Claude {
                return None;
            }
            serde_json::from_value::<types::ContentBlock>(raw.clone()).ok()?
        }
    })
}

fn to_wire_result_part(part: &Part) -> Option<types::ToolResultContent> {
    match part {
        Part::Text(t) => Some(types::ToolResultContent::Text { text: t.clone() }),
        Part::Image { mime, data_base64 } => Some(types::ToolResultContent::Image {
            source: types::ImageSource::Base64 {
                media_type: mime.clone(),
                data: data_base64.clone(),
            },
        }),
        _ => None,
    }
}

// =============================================================================
// Wire -> neutral
// =============================================================================

fn to_neutral_turn(turn: types::AssistantTurn, requested_model: &str) -> AssistantTurn {
    let reasoning = |block: &types::ContentBlock| Part::Reasoning {
        origin: Origin::new(ProviderId::Claude, requested_model),
        raw: serde_json::to_value(block).unwrap_or(Value::Null),
    };

    AssistantTurn {
        parts: turn
            .content
            .iter()
            .map(|b| match b {
                types::ContentBlock::Text { text } => Part::Text(text.clone()),
                types::ContentBlock::ToolUse { id, name, input } => Part::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    args: input.clone(),
                },
                types::ContentBlock::Thinking { .. }
                | types::ContentBlock::RedactedThinking { .. } => reasoning(b),
                types::ContentBlock::Image { source } => match source {
                    types::ImageSource::Base64 { media_type, data } => Part::Image {
                        mime: media_type.clone(),
                        data_base64: data.clone(),
                    },
                },
                // The model never emits tool results; keep it lossless anyway.
                types::ContentBlock::ToolResult { .. } => reasoning(b),
            })
            .collect(),
        stop: turn
            .stop_reason
            .map(|s| to_neutral_stop(s, &turn.stop_details)),
        usage: Usage {
            input_tokens: turn.usage.input_tokens,
            output_tokens: turn.usage.output_tokens,
            cached_input_tokens: turn.usage.cache_read_input_tokens,
        },
        model: turn.model,
    }
}

fn to_neutral_stop(stop: types::StopReason, details: &Option<types::StopDetails>) -> StopReason {
    match stop {
        types::StopReason::EndTurn | types::StopReason::StopSequence => StopReason::EndTurn,
        types::StopReason::MaxTokens => StopReason::MaxTokens,
        types::StopReason::ToolUse => StopReason::ToolUse,
        types::StopReason::PauseTurn => StopReason::Paused,
        types::StopReason::Refusal => StopReason::Refused {
            detail: details.as_ref().and_then(|d| d.explanation.clone()),
        },
        types::StopReason::Unknown => StopReason::Other("unknown".to_string()),
    }
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

/// Tool schemas need no translation for Anthropic — exposed so the shape is
/// testable next to Gemini's translator.
pub fn to_wire_tools(tools: &[ToolSpec]) -> Vec<types::Tool> {
    tools
        .iter()
        .map(|t| types::Tool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn model_role_maps_to_assistant_on_the_wire() {
        let m = to_wire_message(&Turn::model(vec![Part::text("hi")]));
        assert_eq!(m.role, types::Role::Assistant);
    }

    #[test]
    fn tool_results_carry_the_id_and_drop_the_name() {
        // Anthropic pairs purely by id; the neutral name is Gemini's need.
        let block = to_wire_block(&Part::ToolResult {
            id: "c1".into(),
            name: "run_command".into(),
            content: vec![Part::text("done")],
            is_error: false,
        })
        .unwrap();
        match block {
            types::ContentBlock::ToolResult {
                tool_use_id,
                is_error,
                ..
            } => {
                assert_eq!(tool_use_id, "c1");
                assert_eq!(is_error, None, "success must not set is_error");
            }
            other => panic!("expected a tool result, got {other:?}"),
        }
    }

    #[test]
    fn errored_tool_results_set_is_error() {
        let block = to_wire_block(&Part::ToolResult {
            id: "c1".into(),
            name: "run_python".into(),
            content: vec![Part::text("denied")],
            is_error: true,
        })
        .unwrap();
        assert!(matches!(
            block,
            types::ContentBlock::ToolResult {
                is_error: Some(true),
                ..
            }
        ));
    }

    #[test]
    fn screenshots_survive_the_round_trip_into_a_tool_result() {
        let block = to_wire_block(&Part::ToolResult {
            id: "c1".into(),
            name: "screenshot".into(),
            content: vec![Part::text("capture:"), Part::png("QUJD")],
            is_error: false,
        })
        .unwrap();
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""media_type":"image/png""#));
        assert!(json.contains("QUJD"));
    }

    #[test]
    fn reasoning_from_another_provider_is_dropped_not_mangled() {
        let foreign = Part::Reasoning {
            origin: Origin::new(ProviderId::Gemini, "gemini-2.5-pro"),
            raw: json!({"thoughtSignature": "g"}),
        };
        assert!(to_wire_block(&foreign).is_none());
    }

    #[test]
    fn thinking_blocks_round_trip_byte_identically() {
        // The API rejects edited thinking blocks on a continued conversation.
        let original = types::ContentBlock::Thinking {
            thinking: String::new(),
            signature: "sig-abc".into(),
        };
        let neutral = to_neutral_turn(
            types::AssistantTurn {
                content: vec![original.clone()],
                ..Default::default()
            },
            types::DEFAULT_MODEL,
        );
        let back = to_wire_block(&neutral.parts[0]).unwrap();
        assert_eq!(back, original);
    }

    #[test]
    fn reasoning_is_tagged_with_the_requested_model_not_the_resolved_one() {
        // Anthropic answers an alias request with a dated id. Tagging with the
        // reply would make every block look foreign on the very next turn.
        let neutral = to_neutral_turn(
            types::AssistantTurn {
                content: vec![types::ContentBlock::Thinking {
                    thinking: String::new(),
                    signature: "sig".into(),
                }],
                model: "claude-sonnet-5-20250929".to_string(),
                ..Default::default()
            },
            "claude-sonnet-5",
        );
        match &neutral.parts[0] {
            Part::Reasoning { origin, .. } => assert_eq!(origin.model, "claude-sonnet-5"),
            other => panic!("expected reasoning, got {other:?}"),
        }
    }

    #[test]
    fn refusals_carry_their_explanation_into_the_neutral_stop() {
        let stop = to_neutral_stop(
            types::StopReason::Refusal,
            &Some(types::StopDetails {
                category: Some("cyber".into()),
                explanation: Some("declined".into()),
            }),
        );
        assert_eq!(
            stop,
            StopReason::Refused {
                detail: Some("declined".into())
            }
        );
    }

    #[test]
    fn stop_sequence_is_treated_as_a_normal_end_of_turn() {
        assert_eq!(
            to_neutral_stop(types::StopReason::StopSequence, &None),
            StopReason::EndTurn
        );
    }

    #[test]
    fn cache_reads_are_reported_as_cached_input_tokens() {
        let neutral = to_neutral_turn(
            types::AssistantTurn {
                usage: types::Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_input_tokens: 900,
                    cache_creation_input_tokens: 0,
                },
                ..Default::default()
            },
            types::DEFAULT_MODEL,
        );
        assert_eq!(neutral.usage.cached_input_tokens, 900);
    }

    #[test]
    fn tool_specs_pass_through_unchanged() {
        let spec = ToolSpec {
            name: "run_command".into(),
            description: "d".into(),
            input_schema: json!({"type": "object", "properties": {}}),
        };
        let wire = to_wire_tools(std::slice::from_ref(&spec));
        assert_eq!(wire[0].input_schema, spec.input_schema);
    }
}
