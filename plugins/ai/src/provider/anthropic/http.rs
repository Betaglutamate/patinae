//! Raw HTTPS client for the Anthropic Messages API.
//!
//! There is no official Anthropic Rust SDK, so the request/response types and
//! the SSE decoder are ours. Everything here is transport-only and runs on the
//! worker thread — nothing in this module touches viewer state.

use std::sync::OnceLock;
use std::time::Duration;

use super::accum::{TurnAccumulator, TurnEvent};
use super::types;
use crate::provider::{ApiError, ModelInfo};
use crate::sse::SseDecoder;

/// Anthropic Messages endpoint.
///
/// Hardcoded, and deliberately not settings-overridable: the request carries the
/// user's OAuth bearer token, so a configurable base URL would be a way to point
/// that credential at an arbitrary host.
pub const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

/// Model catalogue endpoint. Hardcoded for the same reason as [`MESSAGES_URL`].
pub const MODELS_URL: &str = "https://api.anthropic.com/v1/models?limit=100";

/// Wire version pinned by the Messages API.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Required whenever the request authenticates with an OAuth bearer token
/// rather than an API key. `/v1/messages` rejects OAuth tokens without it.
pub const OAUTH_BETA: &str = "oauth-2025-04-20";

/// Install the ring crypto provider exactly once.
///
/// `reqwest` is built with `rustls-no-provider` across this workspace, so
/// without this every TLS handshake fails. Mirrors
/// `crates/patinae-io/src/fetch.rs`.
pub fn ensure_rustls_crypto_provider() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// User-Agent sent with every request, matching the pattern used for RCSB fetches.
pub fn user_agent() -> String {
    concat!("patinae-ai-plugin/", env!("CARGO_PKG_VERSION")).to_string()
}

/// How long to wait for the TCP+TLS handshake.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// How long the stream may go silent before the request is abandoned.
///
/// This is a per-read inactivity timeout, not a deadline on the whole request:
/// a turn at high effort can legitimately stream for many minutes, and a total
/// timeout would cut it off mid-answer. The API sends comment heartbeats while
/// the model works, so silence this long means the connection is gone.
///
/// Without any timeout, a hung connection wedges the worker thread indefinitely
/// — `Stop` is checked between iterations, not inside a blocking read, so there
/// would be no way to recover short of restarting the viewer.
const READ_TIMEOUT: Duration = Duration::from_secs(180);

/// Build the shared HTTP client. Call [`ensure_rustls_crypto_provider`] first.
pub fn build_client() -> Result<reqwest::Client, ApiError> {
    ensure_rustls_crypto_provider();
    reqwest::Client::builder()
        .user_agent(user_agent())
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()
        .map_err(|e| ApiError::Network(e.to_string()))
}

/// Send one streaming request and accumulate the assistant turn.
///
/// `on_event` is invoked for each [`stream::TurnEvent`] as it arrives, so the
/// panel can render text deltas live. It runs on the worker thread and must not
/// touch viewer state.
pub async fn stream_message(
    client: &reqwest::Client,
    token: &str,
    request: &types::MessagesRequest,
    mut on_event: impl FnMut(TurnEvent),
) -> Result<types::AssistantTurn, ApiError> {
    use futures_util::StreamExt;

    let response = client
        .post(MESSAGES_URL)
        // OAuth tokens go on Authorization: Bearer, never x-api-key, and
        // /v1/messages rejects them without the oauth beta header.
        .bearer_auth(token)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("anthropic-beta", OAUTH_BETA)
        .header("content-type", "application/json")
        .json(request)
        .send()
        .await
        .map_err(|e| ApiError::Network(e.to_string()))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(if status.as_u16() == 401 || status.as_u16() == 403 {
            ApiError::Unauthorized(body)
        } else {
            ApiError::Http {
                status: status.as_u16(),
                body,
            }
        });
    }

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

    Ok(acc.finish())
}

/// Fetch the model catalogue.
///
/// Anthropic publishes ids and display names but no capability metadata, so the
/// entries carry [`ModelInfo`]'s optimistic defaults — which are correct for
/// every model the endpoint currently returns.
pub async fn list_models(
    client: &reqwest::Client,
    token: &str,
) -> Result<Vec<ModelInfo>, ApiError> {
    let response = client
        .get(MODELS_URL)
        .bearer_auth(token)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("anthropic-beta", OAUTH_BETA)
        .send()
        .await
        .map_err(|e| ApiError::Network(e.to_string()))?;

    let status = response.status();
    if !status.is_success() {
        return Err(ApiError::Http {
            status: status.as_u16(),
            body: response.text().await.unwrap_or_default(),
        });
    }

    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|e| ApiError::Network(e.to_string()))?;

    Ok(parse_models(&body))
}

/// Extract the catalogue from a `/v1/models` body. Split out so it is testable
/// without a network round trip.
pub fn parse_models(body: &serde_json::Value) -> Vec<ModelInfo> {
    body.get("data")
        .and_then(|d| d.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_str()?;
                    let label = m
                        .get("display_name")
                        .and_then(|d| d.as_str())
                        .unwrap_or(id)
                        .to_string();
                    Some(ModelInfo::new(id, label))
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn models_are_parsed_with_their_display_names() {
        let body = json!({
            "data": [
                {"type": "model", "id": "claude-sonnet-5", "display_name": "Claude Sonnet 5"},
                {"type": "model", "id": "claude-opus-5", "display_name": "Claude Opus 5"},
            ],
            "has_more": false,
        });
        let models = parse_models(&body);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "claude-sonnet-5");
        assert_eq!(models[0].label, "Claude Sonnet 5");
        assert!(models[0].tools, "Anthropic models all call tools");
    }

    #[test]
    fn a_model_without_a_display_name_falls_back_to_its_id() {
        let body = json!({"data": [{"id": "claude-future-9"}]});
        assert_eq!(parse_models(&body)[0].label, "claude-future-9");
    }

    #[test]
    fn an_unexpected_body_yields_no_models_rather_than_panicking() {
        assert!(parse_models(&json!({})).is_empty());
        assert!(parse_models(&json!({"data": "nonsense"})).is_empty());
        // A malformed entry is skipped without discarding its neighbours.
        let mixed = json!({"data": [{"no_id": true}, {"id": "claude-sonnet-5"}]});
        assert_eq!(parse_models(&mixed).len(), 1);
    }
}
