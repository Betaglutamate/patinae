//! Credential brokering.
//!
//! The plugin stores no secret of its own. Where a vendor ships a CLI that
//! already owns the browser flow, the credential on disk, and refresh — `ant`
//! for Anthropic, `gcloud` for Google — the broker delegates to it and asks for
//! a short-lived credential only when a request is about to go out. OpenRouter
//! ships no CLI, so its broker reads a key the user placed in the environment or
//! in the config directory; it still writes nothing.
//!
//! The vendors want that credential presented differently, so a broker returns a
//! typed [`Credential`] rather than a bare string: sending an API key as a
//! bearer token (or the reverse) fails in ways that look like a bad key.

pub mod ant;
pub mod google;
pub mod openrouter;
pub mod proc;

/// A credential plus how it must be presented on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Credential {
    /// `Authorization: Bearer <token>` — OAuth access tokens.
    Bearer(String),
    /// `x-goog-api-key: <key>` — Google API keys.
    ApiKey(String),
}

/// Whether the user can currently make requests, and what to tell them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthStatus {
    pub signed_in: bool,
    pub message: String,
}

impl AuthStatus {
    pub fn signed_in(message: impl Into<String>) -> Self {
        Self {
            signed_in: true,
            message: message.into(),
        }
    }

    pub fn signed_out(message: impl Into<String>) -> Self {
        Self {
            signed_in: false,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AuthError {
    /// The vendor CLI is not installed; carries an actionable hint.
    ToolMissing(String),
    /// The CLI ran but failed; carries its output.
    Failed(String),
    /// An interactive login outlived its deadline.
    TimedOut(u64),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::ToolMissing(hint) => write!(f, "{hint}"),
            AuthError::Failed(msg) => write!(f, "{msg}"),
            AuthError::TimedOut(secs) => write!(f, "Sign-in timed out after {secs}s."),
        }
    }
}

/// Obtains credentials for one provider.
///
/// Every method blocks on a subprocess and must be called from the worker
/// thread, never from `poll()` or a panel callback.
pub trait AuthBroker: Send + Sync {
    /// A credential for the next request, refreshed if needed.
    fn credential(&self) -> Result<Credential, AuthError>;

    /// Current state, determined by attempting to obtain a credential rather
    /// than by parsing a CLI's human-readable status output.
    fn status(&self) -> AuthStatus {
        match self.credential() {
            Ok(_) => AuthStatus::signed_in("Signed in."),
            Err(e) => AuthStatus::signed_out(e.to_string()),
        }
    }

    /// Run the interactive sign-in, returning any output worth showing.
    fn login(&self) -> Result<String, AuthError>;

    /// Clear the stored credential.
    fn logout(&self) -> Result<(), AuthError>;

    /// Drop any cached credential, forcing a refresh on the next request.
    fn invalidate(&self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_kinds_are_distinct() {
        // Presenting one as the other fails in ways that look like a bad key,
        // so they must not be interchangeable strings.
        assert_ne!(
            Credential::Bearer("x".into()),
            Credential::ApiKey("x".into())
        );
    }

    #[test]
    fn status_constructors_set_the_flag() {
        assert!(AuthStatus::signed_in("ok").signed_in);
        assert!(!AuthStatus::signed_out("nope").signed_in);
    }

    #[test]
    fn errors_render_their_actionable_hint() {
        let e = AuthError::ToolMissing("Install `ant` from …".into());
        assert!(e.to_string().contains("Install"));
        assert!(AuthError::TimedOut(300).to_string().contains("300"));
    }
}
