//! Raw HTTPS client for the OpenRouter API.
//!
//! Transport only, and runs on the worker thread. Shares the SSE framing in
//! [`crate::sse`] with the other providers — which already skips `:`-prefixed
//! comment frames, the shape OpenRouter's `: OPENROUTER PROCESSING` keep-alives
//! take while an upstream model is still thinking.

use super::accum::{TurnAccumulator, TurnEvent};
use super::types;
use crate::provider::{ApiError, ModelInfo, Price};
use crate::sse::SseDecoder;

/// Base for the OpenRouter API.
///
/// Hardcoded, and deliberately not settings-overridable, for the same reason as
/// the other two providers: the request carries the user's key, so a
/// configurable base URL would be a way to point that credential at an arbitrary
/// host. "OpenAI-compatible" makes this an especially tempting setting to add —
/// it is still the wrong one.
pub const API_BASE: &str = "https://openrouter.ai/api/v1";

/// Identifies Patinae on OpenRouter's public leaderboards.
///
/// Both headers are optional, carry no user data, and are the documented way for
/// an application to identify itself.
const REFERER: &str = "https://github.com/Betaglutamate/patinae";
const TITLE: &str = "Patinae";

fn request(
    client: &reqwest::Client,
    method: reqwest::Method,
    path: &str,
    key: &str,
) -> reqwest::RequestBuilder {
    client
        .request(method, format!("{API_BASE}{path}"))
        .bearer_auth(key)
        .header("HTTP-Referer", REFERER)
        .header("X-Title", TITLE)
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
pub async fn stream_chat(
    client: &reqwest::Client,
    key: &str,
    body: &types::ChatRequest,
    mut on_event: impl FnMut(TurnEvent),
) -> Result<types::AssistantTurn, ApiError> {
    use futures_util::StreamExt;

    let response = request(client, reqwest::Method::POST, "/chat/completions", key)
        .header("content-type", "application/json")
        .json(body)
        .send()
        .await
        .map_err(|e| ApiError::Network(e.to_string()))?;

    let response = error_for_status(response).await?;

    let mut decoder = SseDecoder::new();
    let mut acc = TurnAccumulator::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| ApiError::Network(e.to_string()))?;
        for payload in decoder.push(&chunk) {
            for event in acc.push(&payload) {
                on_event(event);
            }
        }
    }

    Ok(acc.finish())
}

/// Fetch the model catalogue.
///
/// Unauthenticated: the endpoint is public, and asking for it before the user
/// has a key means the picker is populated on the very first visit to the panel
/// rather than only after sign-in.
pub async fn list_models(client: &reqwest::Client) -> Result<Vec<ModelInfo>, ApiError> {
    let response = client
        .get(format!("{API_BASE}/models"))
        .header("HTTP-Referer", REFERER)
        .header("X-Title", TITLE)
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

/// Describe the key itself, for the status line.
///
/// Turns "Signed in." into something that answers the question people actually
/// have — which key is this, and is there any credit left on it.
pub async fn describe_key(client: &reqwest::Client, key: &str) -> Result<String, ApiError> {
    let body: serde_json::Value = error_for_status(
        request(client, reqwest::Method::GET, "/key", key)
            .send()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?,
    )
    .await?
    .json()
    .await
    .map_err(|e| ApiError::Network(e.to_string()))?;

    Ok(describe_key_body(&body))
}

/// Render `/key` into a status line. Split out so it is testable offline.
pub fn describe_key_body(body: &serde_json::Value) -> String {
    let data = body.get("data").unwrap_or(body);
    let label = data
        .get("label")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let usage = data.get("usage").and_then(|v| v.as_f64());
    let limit = data.get("limit").and_then(|v| v.as_f64());

    let mut out = String::from("Signed in to OpenRouter");
    if let Some(label) = label {
        out.push_str(&format!(" as `{label}`"));
    }
    match (limit, usage) {
        // A null limit means an unlimited key, which is not the same as zero.
        (Some(limit), Some(usage)) => {
            out.push_str(&format!(" — ${:.2} of ${:.2} left", limit - usage, limit))
        }
        (None, Some(usage)) => out.push_str(&format!(" — ${usage:.2} used")),
        _ => {}
    }
    out.push('.');
    out
}

/// Extract the catalogue from a `/models` body.
///
/// Capabilities come from `supported_parameters` and `architecture`, which is
/// the whole reason this endpoint is worth calling: on OpenRouter these vary per
/// model, and guessing them from the vendor prefix would be wrong more often
/// than not.
pub fn parse_models(body: &serde_json::Value) -> Vec<ModelInfo> {
    let Some(entries) = body.get("data").and_then(|d| d.as_array()) else {
        return Vec::new();
    };

    let mut models: Vec<ModelInfo> = entries
        .iter()
        .filter_map(|m| {
            let id = m.get("id")?.as_str()?.to_string();
            let label = m
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or(&id)
                .to_string();

            let has_param = |name: &str| {
                m.get("supported_parameters")
                    .and_then(|p| p.as_array())
                    .map(|params| params.iter().any(|x| x.as_str() == Some(name)))
                    .unwrap_or(false)
            };

            let images = m
                .get("architecture")
                .and_then(|a| a.get("input_modalities"))
                .and_then(|v| v.as_array())
                .map(|kinds| kinds.iter().any(|k| k.as_str() == Some("image")))
                .unwrap_or(false);

            Some(ModelInfo {
                id,
                label,
                context_length: m
                    .get("context_length")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32),
                tools: has_param("tools"),
                images,
                reasoning: has_param("reasoning") || has_param("include_reasoning"),
                price: parse_price(m.get("pricing")),
            })
        })
        .collect();

    // Cheapest-looking first is meaningless across hundreds of models; a stable
    // alphabetical order at least makes the filtered picker predictable.
    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

/// Prices are quoted per token as decimal *strings*, so they arrive as e.g.
/// `"0.000003"` and are scaled to the per-million-token figures people quote.
fn parse_price(pricing: Option<&serde_json::Value>) -> Option<Price> {
    let pricing = pricing?;
    let field = |name: &str| -> Option<f64> {
        let value = pricing.get(name)?;
        value
            .as_str()
            .and_then(|s| s.parse::<f64>().ok())
            .or_else(|| value.as_f64())
    };
    let prompt = field("prompt")?;
    let completion = field("completion")?;
    Some(Price {
        prompt: prompt * 1_000_000.0,
        completion: completion * 1_000_000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_catalogue() -> serde_json::Value {
        json!({"data": [
            {
                "id": "anthropic/claude-sonnet-5",
                "name": "Anthropic: Claude Sonnet 5",
                "context_length": 200_000,
                "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                "supported_parameters": ["tools", "tool_choice", "reasoning", "max_tokens"],
            },
            {
                "id": "some/completion-only",
                "name": "Completion Only",
                "context_length": 8_192,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
                "supported_parameters": ["max_tokens", "temperature"],
            },
        ]})
    }

    #[test]
    fn capabilities_are_read_from_the_catalogue_not_guessed() {
        let models = parse_models(&sample_catalogue());
        let sonnet = models
            .iter()
            .find(|m| m.id.starts_with("anthropic/"))
            .unwrap();
        assert!(sonnet.tools);
        assert!(sonnet.images);
        assert!(sonnet.reasoning);
        assert_eq!(sonnet.context_length, Some(200_000));

        let plain = models.iter().find(|m| m.id.starts_with("some/")).unwrap();
        assert!(!plain.tools, "no `tools` parameter means no tool calling");
        assert!(!plain.images, "text-only input modalities means no vision");
        assert!(!plain.reasoning);
    }

    #[test]
    fn prices_are_scaled_from_per_token_strings_to_per_million() {
        let models = parse_models(&sample_catalogue());
        let price = models[0].price.unwrap();
        assert!((price.prompt - 3.0).abs() < 1e-9);
        assert!((price.completion - 15.0).abs() < 1e-9);
    }

    #[test]
    fn a_model_with_no_published_price_still_appears() {
        let body = json!({"data": [{"id": "x/y", "supported_parameters": ["tools"]}]});
        let models = parse_models(&body);
        assert_eq!(models.len(), 1);
        assert!(models[0].price.is_none());
        assert!(models[0].context_length.is_none());
    }

    #[test]
    fn free_models_priced_at_zero_are_not_mistaken_for_unpriced() {
        let body = json!({"data": [{
            "id": "x/y:free",
            "pricing": {"prompt": "0", "completion": "0"},
            "supported_parameters": ["tools"],
        }]});
        let price = parse_models(&body)[0].price.unwrap();
        assert_eq!(price.prompt, 0.0);
    }

    #[test]
    fn the_catalogue_comes_back_in_a_stable_order() {
        let body = json!({"data": [
            {"id": "z/model", "supported_parameters": ["tools"]},
            {"id": "a/model", "supported_parameters": ["tools"]},
        ]});
        let ids: Vec<String> = parse_models(&body).into_iter().map(|m| m.id).collect();
        assert_eq!(ids, ["a/model", "z/model"]);
    }

    #[test]
    fn an_unexpected_body_yields_no_models_rather_than_panicking() {
        assert!(parse_models(&json!({})).is_empty());
        assert!(parse_models(&json!({"data": "nonsense"})).is_empty());
        assert_eq!(
            parse_models(&json!({"data": [{"no_id": 1}, {"id": "a/b"}]})).len(),
            1
        );
    }

    #[test]
    fn the_key_status_line_reports_the_label_and_remaining_credit() {
        let body = json!({"data": {"label": "patinae", "usage": 7.6, "limit": 20.0}});
        let line = describe_key_body(&body);
        assert!(line.contains("patinae"));
        assert!(line.contains("$12.40 of $20.00 left"));
    }

    #[test]
    fn an_unlimited_key_reports_usage_rather_than_a_bogus_balance() {
        // A null limit is unlimited, not zero — reporting "-$7.60 left" would
        // be actively misleading.
        let body = json!({"data": {"label": "patinae", "usage": 7.6, "limit": null}});
        let line = describe_key_body(&body);
        assert!(line.contains("$7.60 used"));
        assert!(!line.contains("left"));
    }

    #[test]
    fn a_key_with_no_metadata_still_reports_being_signed_in() {
        assert_eq!(
            describe_key_body(&json!({"data": {}})),
            "Signed in to OpenRouter."
        );
    }
}
