//! Raw HTTPS client for the Anthropic Messages API.
//!
//! There is no official Anthropic Rust SDK, so the request/response types and
//! the SSE decoder are ours. Everything here is transport-only and runs on the
//! worker thread — nothing in this module touches viewer state.

pub mod stream;
pub mod types;

use std::sync::OnceLock;
use std::time::Duration;

/// Anthropic Messages endpoint.
///
/// Hardcoded, and deliberately not settings-overridable: the request carries the
/// user's OAuth bearer token, so a configurable base URL would be a way to point
/// that credential at an arbitrary host.
pub const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

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
    concat!("patinae-claude-plugin/", env!("CARGO_PKG_VERSION")).to_string()
}

/// A failure while calling the Messages API.
#[derive(Debug, Clone)]
pub enum ApiError {
    /// Transport-level failure (DNS, TLS, connection reset).
    Network(String),
    /// The token was rejected. The caller should invalidate the cached token
    /// and retry once before surfacing this.
    Unauthorized(String),
    /// Any other non-2xx response, with the body for diagnosis.
    Http { status: u16, body: String },
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
    mut on_event: impl FnMut(stream::TurnEvent),
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

    let mut decoder = stream::SseDecoder::new();
    let mut acc = stream::TurnAccumulator::new();
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
