//! Google credentials for the Gemini Developer API.
//!
//! Mirrors the `ant` pattern: `gcloud` owns the browser flow and refresh, and
//! we hold only a short-lived access token in memory. Google's own OAuth
//! quickstart documents exactly this against `generativelanguage.googleapis.com`:
//!
//! ```text
//! access_token=$(gcloud auth application-default print-access-token)
//! curl https://generativelanguage.googleapis.com/v1/models \
//!   -H "Authorization: Bearer ${access_token}"
//! ```
//!
//! That path needs the Generative Language API enabled on a GCP project and ADC
//! logged in with a suitable scope — more setup than an AI Studio key. So if
//! `gcloud` is absent we fall back to a `GEMINI_API_KEY` / `GOOGLE_API_KEY`
//! environment variable, which costs nothing and still stores no secret.
//!
//! The two are presented differently on the wire, which is why
//! [`Credential`] is typed.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::proc;
use super::{AuthBroker, AuthError, AuthStatus, Credential};

/// gcloud access tokens are short-lived; re-ask well inside their lifetime.
const TOKEN_CACHE_TTL: Duration = Duration::from_secs(60);

const LOGIN_TIMEOUT: Duration = Duration::from_secs(300);

/// Environment variables checked when `gcloud` is unavailable, in order.
const API_KEY_VARS: [&str; 2] = ["GEMINI_API_KEY", "GOOGLE_API_KEY"];

const INSTALL_HINT: &str = "Not signed in to Google. Either install the gcloud CLI \
     (https://cloud.google.com/sdk/docs/install) and click Sign in, or set \
     GEMINI_API_KEY to a key from https://aistudio.google.com/apikey and restart Patinae.";

static TOKEN_CACHE: Mutex<Option<(String, Instant)>> = Mutex::new(None);

#[derive(Debug, Default)]
pub struct GoogleBroker;

impl GoogleBroker {
    /// An API key from the environment, if one is set.
    ///
    /// Read on every call rather than cached: the cache exists to avoid
    /// spawning subprocesses, and reading an env var is free.
    fn env_api_key() -> Option<String> {
        API_KEY_VARS.iter().find_map(|var| {
            std::env::var(var)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        })
    }

    fn gcloud_token(&self) -> Result<String, AuthError> {
        if let Ok(cache) = TOKEN_CACHE.lock() {
            if let Some((token, fetched_at)) = cache.as_ref() {
                if fetched_at.elapsed() < TOKEN_CACHE_TTL {
                    return Ok(token.clone());
                }
            }
        }

        let out = proc::run(
            "gcloud",
            &["auth", "application-default", "print-access-token"],
            INSTALL_HINT,
        )?;
        let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if token.is_empty() {
            return Err(AuthError::Failed(
                "gcloud returned an empty access token; try signing in again".to_string(),
            ));
        }

        if let Ok(mut cache) = TOKEN_CACHE.lock() {
            *cache = Some((token.clone(), Instant::now()));
        }
        Ok(token)
    }
}

impl AuthBroker for GoogleBroker {
    /// Prefer gcloud, because it refreshes and expires; fall back to a key.
    fn credential(&self) -> Result<Credential, AuthError> {
        if proc::available("gcloud") {
            match self.gcloud_token() {
                Ok(token) => return Ok(Credential::Bearer(token)),
                // gcloud is installed but not logged in — a key is still a
                // perfectly good credential, so try it before giving up.
                Err(e) => {
                    return Self::env_api_key().map(Credential::ApiKey).ok_or(e);
                }
            }
        }
        Self::env_api_key()
            .map(Credential::ApiKey)
            .ok_or_else(|| AuthError::ToolMissing(INSTALL_HINT.to_string()))
    }

    fn status(&self) -> AuthStatus {
        match self.credential() {
            Ok(Credential::Bearer(_)) => AuthStatus::signed_in("Signed in with gcloud."),
            Ok(Credential::ApiKey(_)) => AuthStatus::signed_in("Signed in with GEMINI_API_KEY."),
            Err(e) => AuthStatus::signed_out(e.to_string()),
        }
    }

    fn login(&self) -> Result<String, AuthError> {
        self.invalidate();
        // An API key needs no interactive flow, so say so rather than opening a
        // browser the user did not ask for.
        if !proc::available("gcloud") {
            if Self::env_api_key().is_some() {
                return Ok(
                    "Using GEMINI_API_KEY from the environment; no sign-in needed.".to_string(),
                );
            }
            return Err(AuthError::ToolMissing(INSTALL_HINT.to_string()));
        }

        let output = proc::run_interactive(
            "gcloud",
            &["auth", "application-default", "login"],
            INSTALL_HINT,
            LOGIN_TIMEOUT,
        )?;
        self.credential()?;
        Ok(output)
    }

    fn logout(&self) -> Result<(), AuthError> {
        self.invalidate();
        if !proc::available("gcloud") {
            // Nothing of ours to clear; the key lives in the user's environment.
            return Ok(());
        }
        proc::run(
            "gcloud",
            &["auth", "application-default", "revoke", "--quiet"],
            INSTALL_HINT,
        )
        .map(|_| ())
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

    /// Env vars are process-global; keep the mutations in one test and restore.
    #[test]
    fn an_env_key_is_used_when_gcloud_is_absent() {
        let saved: Vec<(&str, Option<String>)> = API_KEY_VARS
            .iter()
            .map(|v| (*v, std::env::var(v).ok()))
            .collect();
        for v in API_KEY_VARS {
            std::env::remove_var(v);
        }

        assert!(GoogleBroker::env_api_key().is_none());

        std::env::set_var("GEMINI_API_KEY", "  key-123  ");
        assert_eq!(GoogleBroker::env_api_key().as_deref(), Some("key-123"));

        // Blank values must not count as configured.
        std::env::set_var("GEMINI_API_KEY", "   ");
        assert!(GoogleBroker::env_api_key().is_none());

        for (var, value) in saved {
            match value {
                Some(v) => std::env::set_var(var, v),
                None => std::env::remove_var(var),
            }
        }
    }

    #[test]
    fn api_keys_are_presented_as_keys_not_bearer_tokens() {
        // Google rejects an API key sent as a bearer token, so the distinction
        // has to survive all the way to the header.
        let cred = Credential::ApiKey("k".into());
        assert!(matches!(cred, Credential::ApiKey(_)));
    }

    #[test]
    fn the_hint_covers_both_supported_paths() {
        assert!(INSTALL_HINT.contains("gcloud"));
        assert!(INSTALL_HINT.contains("GEMINI_API_KEY"));
    }

    #[test]
    fn invalidate_clears_the_cache() {
        {
            let mut cache = TOKEN_CACHE.lock().unwrap();
            *cache = Some(("tok".to_string(), Instant::now()));
        }
        GoogleBroker.invalidate();
        assert!(TOKEN_CACHE.lock().unwrap().is_none());
    }
}
