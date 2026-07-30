//! Raw HTTPS client for the Gemini Developer API.
//!
//! Transport only, and runs on the worker thread — nothing here touches viewer
//! state. Shares the SSE framing in [`crate::sse`] with the other providers;
//! what the frames *mean* lives in [`super::accum`].

use super::accum::{TurnAccumulator, TurnEvent};
use super::types;
use crate::auth::Credential;
use crate::provider::{ApiError, ModelInfo};
use crate::sse::SseDecoder;

/// Base for the Gemini Developer API.
///
/// Hardcoded, and deliberately not settings-overridable, for the same reason as
/// the Anthropic endpoint: the request carries the user's access token, so a
/// configurable base URL would be a way to point that credential at an
/// arbitrary host.
pub const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Characters permitted in a model id.
///
/// The model comes from a user setting and is interpolated into the request
/// *path*, so it is validated rather than escaped: `../` in a model name would
/// otherwise choose a different endpoint on the same host, and a name
/// containing `?` or `#` would truncate the path and drop `alt=sse`.
fn is_valid_model_id(model: &str) -> bool {
    !model.is_empty()
        && model
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
}

fn model_path(model: &str, method: &str) -> Result<String, ApiError> {
    if !is_valid_model_id(model) {
        return Err(ApiError::Network(format!(
            "`{model}` is not a valid Gemini model name; expected something like `gemini-2.5-pro`"
        )));
    }
    Ok(format!("{API_BASE}/models/{model}:{method}"))
}

/// Attach whichever credential shape the Google broker produced.
///
/// A gcloud access token goes on `Authorization: Bearer`; an AI Studio API key
/// goes on `x-goog-api-key`. Presenting either as the other fails in a way that
/// looks like a bad key, which is why the credential is typed all the way here.
fn authenticate(
    request: reqwest::RequestBuilder,
    credential: &Credential,
) -> reqwest::RequestBuilder {
    match credential {
        Credential::Bearer(token) => request.bearer_auth(token),
        Credential::ApiKey(key) => request.header("x-goog-api-key", key),
    }
}

async fn error_for_status(response: reqwest::Response) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    Err(if matches!(status.as_u16(), 401 | 403) {
        ApiError::Unauthorized(body)
    } else {
        ApiError::Http {
            status: status.as_u16(),
            body,
        }
    })
}

/// Send one streaming request and accumulate the assistant turn.
pub async fn stream_generate(
    client: &reqwest::Client,
    credential: &Credential,
    model: &str,
    request: &types::GenerateContentRequest,
    mut on_event: impl FnMut(TurnEvent),
) -> Result<types::AssistantTurn, ApiError> {
    use futures_util::StreamExt;

    // Without `alt=sse` the endpoint streams a JSON *array* rather than SSE
    // frames, which the shared decoder cannot read.
    let url = format!("{}?alt=sse", model_path(model, "streamGenerateContent")?);

    let response = authenticate(client.post(url), credential)
        .header("content-type", "application/json")
        .json(request)
        .send()
        .await
        .map_err(|e| ApiError::Network(e.to_string()))?;

    let response = error_for_status(response).await?;

    let mut decoder = SseDecoder::new();
    let mut acc = TurnAccumulator::new();
    let mut body = response.bytes_stream();

    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|e| ApiError::Network(e.to_string()))?;
        for payload in decoder.push(&chunk) {
            for event in acc.push(&payload) {
                on_event(event);
            }
        }
    }

    on_event(TurnEvent::Done);
    Ok(acc.finish())
}

/// Fetch the model catalogue.
pub async fn list_models(
    client: &reqwest::Client,
    credential: &Credential,
) -> Result<Vec<ModelInfo>, ApiError> {
    let url = format!("{API_BASE}/models?pageSize=200");
    let response = authenticate(client.get(url), credential)
        .send()
        .await
        .map_err(|e| ApiError::Network(e.to_string()))?;

    let body: serde_json::Value = error_for_status(response)
        .await?
        .json()
        .await
        .map_err(|e| ApiError::Network(e.to_string()))?;

    Ok(parse_models(&body))
}

/// Extract the catalogue from a `/v1beta/models` body.
///
/// Filtered to models that can actually back the agent: anything not offering
/// `generateContent` (embedding and token-counting models) would fail on the
/// first turn, and offering it in the picker is a trap rather than a choice.
pub fn parse_models(body: &serde_json::Value) -> Vec<ModelInfo> {
    let Some(entries) = body.get("models").and_then(|m| m.as_array()) else {
        return Vec::new();
    };

    entries
        .iter()
        .filter_map(|m| {
            let supports_generate = m
                .get("supportedGenerationMethods")
                .and_then(|v| v.as_array())
                .map(|methods| {
                    methods
                        .iter()
                        .any(|x| x.as_str() == Some("generateContent"))
                })
                .unwrap_or(false);
            if !supports_generate {
                return None;
            }

            // `name` is fully qualified as `models/gemini-2.5-pro`, but the
            // request path adds the prefix itself.
            let id = m
                .get("name")?
                .as_str()?
                .strip_prefix("models/")?
                .to_string();
            let label = m
                .get("displayName")
                .and_then(|d| d.as_str())
                .unwrap_or(&id)
                .to_string();

            let mut info = ModelInfo::new(id, label);
            if let Some(limit) = m.get("inputTokenLimit").and_then(|v| v.as_u64()) {
                info.context_length = Some(limit as u32);
            }
            // The catalogue does not publish tool or vision support, so the
            // optimistic defaults stand; `thinking` is published when present.
            if let Some(thinking) = m.get("thinking").and_then(|v| v.as_bool()) {
                info.reasoning = thinking;
            }
            Some(info)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_well_formed_model_builds_the_streaming_path() {
        let url = model_path("gemini-2.5-pro", "streamGenerateContent").unwrap();
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent"
        );
    }

    #[test]
    fn a_model_name_cannot_redirect_the_request() {
        // The model is a user setting spliced into the path. Traversal would
        // reach a different endpoint; `?` or `#` would truncate away alt=sse.
        for hostile in [
            "../../v1/other",
            "gemini-2.5-pro?key=leak",
            "gemini-2.5-pro#frag",
            "gemini 2.5 pro",
            "gemini/../../x",
            "",
        ] {
            assert!(
                model_path(hostile, "streamGenerateContent").is_err(),
                "{hostile:?} should be rejected"
            );
        }
    }

    #[test]
    fn ordinary_model_names_are_accepted() {
        for ok in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini_3.0", "abc123"] {
            assert!(is_valid_model_id(ok), "{ok} should be accepted");
        }
    }

    #[test]
    fn only_models_that_can_generate_content_are_offered() {
        let body = json!({"models": [
            {
                "name": "models/gemini-2.5-pro",
                "displayName": "Gemini 2.5 Pro",
                "inputTokenLimit": 1048576,
                "supportedGenerationMethods": ["generateContent", "countTokens"],
            },
            {
                "name": "models/text-embedding-004",
                "displayName": "Text Embedding",
                "supportedGenerationMethods": ["embedContent"],
            },
        ]});

        let models = parse_models(&body);
        assert_eq!(models.len(), 1);
        assert_eq!(
            models[0].id, "gemini-2.5-pro",
            "the models/ prefix is stripped"
        );
        assert_eq!(models[0].context_length, Some(1_048_576));
    }

    #[test]
    fn published_thinking_support_overrides_the_optimistic_default() {
        let body = json!({"models": [{
            "name": "models/gemini-2.5-flash-lite",
            "supportedGenerationMethods": ["generateContent"],
            "thinking": false,
        }]});
        assert!(!parse_models(&body)[0].reasoning);
    }

    #[test]
    fn an_unexpected_body_yields_no_models_rather_than_panicking() {
        assert!(parse_models(&json!({})).is_empty());
        assert!(parse_models(&json!({"models": "nonsense"})).is_empty());
        // A model missing supportedGenerationMethods is skipped, not assumed.
        assert!(parse_models(&json!({"models": [{"name": "models/x"}]})).is_empty());
    }
}
