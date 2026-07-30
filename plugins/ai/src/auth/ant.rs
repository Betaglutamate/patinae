//! Anthropic credentials, brokered through the official `ant` CLI.
//!
//! `ant` owns the browser OAuth flow, the credential profile on disk, and token
//! refresh. We only ask it for a short-lived access token and hold that in
//! memory; nothing is written to `~/.patinae`.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::proc;
use super::{AuthBroker, AuthError, Credential};

/// How long a fetched token is reused before asking `ant` again.
///
/// `ant` refreshes on demand, so this only avoids spawning a subprocess for
/// every request in a long agent loop.
const TOKEN_CACHE_TTL: Duration = Duration::from_secs(60);

const LOGIN_TIMEOUT: Duration = Duration::from_secs(300);

const INSTALL_HINT: &str = "The `ant` CLI is not installed. Install it from \
     https://github.com/anthropics/anthropic-cli/releases (or \
     `brew install anthropics/tap/ant`), then sign in.";

static TOKEN_CACHE: Mutex<Option<(String, Instant)>> = Mutex::new(None);

#[derive(Debug, Default)]
pub struct AntBroker;

impl AuthBroker for AntBroker {
    /// Anthropic OAuth tokens go on `Authorization: Bearer`, never `x-api-key`.
    ///
    /// Note `--access-token` is required: bare `print-credentials` emits a JSON
    /// object, and pasting that into a header yields an opaque auth failure.
    fn credential(&self) -> Result<Credential, AuthError> {
        if let Ok(cache) = TOKEN_CACHE.lock() {
            if let Some((token, fetched_at)) = cache.as_ref() {
                if fetched_at.elapsed() < TOKEN_CACHE_TTL {
                    return Ok(Credential::Bearer(token.clone()));
                }
            }
        }

        let out = proc::run_secret(
            "ant",
            &["auth", "print-credentials", "--access-token"],
            INSTALL_HINT,
        )?;
        let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if token.is_empty() {
            return Err(AuthError::Failed(
                "ant returned an empty access token; try signing in again".to_string(),
            ));
        }

        if let Ok(mut cache) = TOKEN_CACHE.lock() {
            *cache = Some((token.clone(), Instant::now()));
        }
        Ok(Credential::Bearer(token))
    }

    fn login(&self) -> Result<String, AuthError> {
        self.invalidate();
        let output = proc::run_interactive("ant", &["auth", "login"], INSTALL_HINT, LOGIN_TIMEOUT)?;
        // Prove the login produced a usable credential rather than trusting the
        // exit code alone.
        self.credential()?;
        Ok(output)
    }

    fn logout(&self) -> Result<(), AuthError> {
        self.invalidate();
        proc::run("ant", &["auth", "logout"], INSTALL_HINT).map(|_| ())
    }

    fn invalidate(&self) {
        if let Ok(mut cache) = TOKEN_CACHE.lock() {
            *cache = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_presented_as_bearer_not_api_key() {
        AntBroker.invalidate();
        {
            let mut cache = TOKEN_CACHE.lock().unwrap();
            *cache = Some(("tok".to_string(), Instant::now()));
        }
        assert_eq!(
            AntBroker.credential().unwrap(),
            Credential::Bearer("tok".into())
        );
        AntBroker.invalidate();
    }

    #[test]
    fn invalidate_clears_the_cache() {
        {
            let mut cache = TOKEN_CACHE.lock().unwrap();
            *cache = Some(("tok".to_string(), Instant::now()));
        }
        AntBroker.invalidate();
        assert!(TOKEN_CACHE.lock().unwrap().is_none());
    }

    #[test]
    fn expired_cache_entries_are_not_reused() {
        AntBroker.invalidate();
        {
            let mut cache = TOKEN_CACHE.lock().unwrap();
            let stale = Instant::now() - TOKEN_CACHE_TTL - Duration::from_secs(1);
            *cache = Some(("stale".to_string(), stale));
        }
        // Falls through to `ant`, which is absent in CI — the point is that the
        // stale value is not returned.
        assert!(!matches!(
            AntBroker.credential(),
            Ok(Credential::Bearer(t)) if t == "stale"
        ));
        AntBroker.invalidate();
    }

    #[test]
    fn install_hint_names_the_binary_and_where_to_get_it() {
        assert!(INSTALL_HINT.contains("ant"));
        assert!(INSTALL_HINT.contains("http"));
    }
}
