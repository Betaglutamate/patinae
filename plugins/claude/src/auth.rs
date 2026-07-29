//! Authentication, brokered through Anthropic's official `ant` CLI.
//!
//! The plugin deliberately stores no secret of its own. `ant` owns the browser
//! OAuth flow, the credential profile on disk, and token refresh; we only ask it
//! for a short-lived access token when a request is about to go out, and hold
//! that token in memory.
//!
//! Every function here blocks on a subprocess and must be called from the worker
//! thread, never from `poll()` or a panel callback.

use std::io;
use std::process::{Command, Output, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long a fetched access token is reused before asking `ant` again.
///
/// `ant` refreshes on demand, so this is only here to avoid spawning a
/// subprocess for every request in a long agent loop.
const TOKEN_CACHE_TTL: Duration = Duration::from_secs(60);

/// How long to wait for an interactive `ant auth login` before giving up.
const LOGIN_TIMEOUT: Duration = Duration::from_secs(300);

/// Whether the user can currently make API calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthState {
    /// The `ant` binary was not found on `PATH`.
    CliMissing,
    /// `ant` is installed but has no usable credential.
    SignedOut,
    /// A token was obtained successfully.
    SignedIn,
}

impl AuthState {
    /// Message to show in the panel for this state.
    pub fn describe(&self) -> &'static str {
        match self {
            AuthState::CliMissing => {
                "The `ant` CLI is not installed. Install it from \
                 https://github.com/anthropics/anthropic-cli/releases \
                 (or `brew install anthropics/tap/ant`), then sign in."
            }
            AuthState::SignedOut => {
                "Not signed in. Click Sign in to authorize with your Claude account."
            }
            AuthState::SignedIn => "Signed in.",
        }
    }
}

/// What went wrong while talking to `ant`.
#[derive(Debug, Clone)]
pub enum AuthError {
    /// `ant` is not installed or not on `PATH`.
    CliMissing,
    /// `ant` ran but exited non-zero; carries its stderr.
    Failed(String),
    /// The login subprocess outlived [`LOGIN_TIMEOUT`].
    LoginTimedOut,
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::CliMissing => write!(f, "{}", AuthState::CliMissing.describe()),
            AuthError::Failed(msg) => write!(f, "ant failed: {msg}"),
            AuthError::LoginTimedOut => {
                write!(f, "Sign-in timed out after {}s.", LOGIN_TIMEOUT.as_secs())
            }
        }
    }
}

/// Cached access token and the moment it was fetched.
static TOKEN_CACHE: Mutex<Option<(String, Instant)>> = Mutex::new(None);

/// Fetch a short-lived OAuth access token.
///
/// The token goes on `Authorization: Bearer`, **not** `x-api-key`, and the
/// request must also carry `anthropic-beta: oauth-2025-04-20`.
///
/// Note the `--access-token` flag is required: bare `print-credentials` emits a
/// JSON object, not the token, and pasting that into a header yields an opaque
/// auth failure.
pub fn access_token() -> Result<String, AuthError> {
    if let Ok(cache) = TOKEN_CACHE.lock() {
        if let Some((token, fetched_at)) = cache.as_ref() {
            if fetched_at.elapsed() < TOKEN_CACHE_TTL {
                return Ok(token.clone());
            }
        }
    }

    let out = run_ant(&["auth", "print-credentials", "--access-token"])?;
    let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if token.is_empty() {
        return Err(AuthError::Failed(
            "ant returned an empty access token; try signing in again".to_string(),
        ));
    }

    if let Ok(mut cache) = TOKEN_CACHE.lock() {
        *cache = Some((token.clone(), Instant::now()));
    }
    Ok(token)
}

/// Current auth state.
///
/// Determined by attempting a token fetch rather than parsing `ant auth status`
/// output — the human-readable status text is not a stable interface, whereas
/// "can we actually get a token" is precisely the question we need answered.
pub fn state() -> AuthState {
    match access_token() {
        Ok(_) => AuthState::SignedIn,
        Err(AuthError::CliMissing) => AuthState::CliMissing,
        Err(_) => AuthState::SignedOut,
    }
}

/// Run the interactive browser sign-in.
///
/// Blocks until `ant` exits, the user abandons the flow, or [`LOGIN_TIMEOUT`]
/// elapses. Output is captured and returned so the panel can surface a URL (in
/// `--no-browser` mode) or an error — a GUI user has no terminal to read.
pub fn login() -> Result<String, AuthError> {
    invalidate_cache();

    let mut child = Command::new("ant")
        .args(["auth", "login"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(map_spawn_error)?;

    let deadline = Instant::now() + LOGIN_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(AuthError::LoginTimedOut);
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => return Err(AuthError::Failed(e.to_string())),
        }
    }

    let out = child
        .wait_with_output()
        .map_err(|e| AuthError::Failed(e.to_string()))?;
    check_status(&out)?;

    // Prove the login actually produced a usable credential rather than
    // trusting the exit code alone.
    access_token()?;
    Ok(combined_output(&out))
}

/// Clear the active `ant` profile.
pub fn logout() -> Result<(), AuthError> {
    invalidate_cache();
    run_ant(&["auth", "logout"]).map(|_| ())
}

/// Drop any cached token, forcing the next request to re-fetch.
pub fn invalidate_cache() {
    if let Ok(mut cache) = TOKEN_CACHE.lock() {
        *cache = None;
    }
}

/// Run `ant` with the given arguments, mapping failures onto [`AuthError`].
fn run_ant(args: &[&str]) -> Result<Output, AuthError> {
    let out = Command::new("ant")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(map_spawn_error)?;
    check_status(&out)?;
    Ok(out)
}

fn map_spawn_error(e: io::Error) -> AuthError {
    if e.kind() == io::ErrorKind::NotFound {
        AuthError::CliMissing
    } else {
        AuthError::Failed(e.to_string())
    }
}

fn check_status(out: &Output) -> Result<(), AuthError> {
    if out.status.success() {
        return Ok(());
    }
    let msg = combined_output(out);
    Err(AuthError::Failed(if msg.is_empty() {
        format!("exited with {}", out.status)
    } else {
        msg
    }))
}

fn combined_output(out: &Output) -> String {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let mut parts: Vec<&str> = Vec::new();
    if !stdout.trim().is_empty() {
        parts.push(stdout.trim());
    }
    if !stderr.trim().is_empty() {
        parts.push(stderr.trim());
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_binary_is_reported_as_cli_missing() {
        let e = map_spawn_error(io::Error::new(io::ErrorKind::NotFound, "no such file"));
        assert!(matches!(e, AuthError::CliMissing));
    }

    #[test]
    fn other_spawn_errors_are_not_cli_missing() {
        let e = map_spawn_error(io::Error::new(io::ErrorKind::PermissionDenied, "denied"));
        assert!(matches!(e, AuthError::Failed(_)));
    }

    #[test]
    fn cli_missing_message_names_the_binary() {
        assert!(AuthState::CliMissing.describe().contains("ant"));
    }

    #[test]
    fn cache_round_trips_and_invalidates() {
        invalidate_cache();
        {
            let mut cache = TOKEN_CACHE.lock().unwrap();
            *cache = Some(("tok".to_string(), Instant::now()));
        }
        assert_eq!(access_token().unwrap(), "tok");
        invalidate_cache();
        assert!(TOKEN_CACHE.lock().unwrap().is_none());
    }

    #[test]
    fn expired_cache_entries_are_not_reused() {
        invalidate_cache();
        {
            let mut cache = TOKEN_CACHE.lock().unwrap();
            let stale = Instant::now() - TOKEN_CACHE_TTL - Duration::from_secs(1);
            *cache = Some(("stale".to_string(), stale));
        }
        // Falls through to `ant`, which is absent in CI — the point is that the
        // stale value is not returned.
        assert!(!matches!(access_token(), Ok(t) if t == "stale"));
    }
}
