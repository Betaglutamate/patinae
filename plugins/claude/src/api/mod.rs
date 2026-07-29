//! Raw HTTPS client for the Anthropic Messages API.
//!
//! There is no official Anthropic Rust SDK, so the request/response types and
//! the SSE decoder are ours. Everything here is transport-only and runs on the
//! worker thread — nothing in this module touches viewer state.

pub mod stream;

use std::sync::OnceLock;

/// Anthropic Messages endpoint.
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
