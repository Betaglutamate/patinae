//! OpenRouter credentials.
//!
//! The third auth shape in the plugin, and the odd one out. `ant` and `gcloud`
//! are OAuth CLIs that own a browser flow, a credential on disk, and refresh;
//! OpenRouter has no CLI at all, only a long-lived API key.
//!
//! The plugin's invariant is that it stores no secret of its own, and that holds
//! here: the key comes from `OPENROUTER_API_KEY`, or from a file the *user*
//! creates at `<config_dir>/openrouter.key`. Nothing writes it. The file exists
//! because an environment variable alone is not enough — a GUI-launched app
//! bundle on macOS inherits no shell environment, so the env var would be
//! invisible exactly where Patinae is most often started from.
//!
//! Unlike an OAuth token there is nothing to refresh, so there is no token cache
//! here — reading an env var or a small file per request is free next to the
//! request itself, and it means a key added while Patinae is running takes
//! effect without a restart.

use std::path::PathBuf;

use super::{AuthBroker, AuthError, AuthStatus, Credential};

/// Environment variable checked first.
pub const API_KEY_VAR: &str = "OPENROUTER_API_KEY";

/// Name of the key file inside the Patinae config directory.
pub const KEY_FILE_NAME: &str = "openrouter.key";

/// Where to get a key. Shown whenever no credential can be found.
const SETUP_HINT: &str = "Not signed in to OpenRouter. Create a key at \
     https://openrouter.ai/keys, then either set OPENROUTER_API_KEY or save it \
     to ~/.patinae/openrouter.key (the file should contain the key and nothing else).";

pub fn key_path() -> PathBuf {
    patinae_settings::paths::config_dir().join(KEY_FILE_NAME)
}

/// Pull a key out of raw file contents.
///
/// Takes the first non-blank, non-comment line so a key file that picked up a
/// trailing newline, a stray comment, or a `KEY=` style export still works —
/// people assemble this file by hand, and a silent auth failure over a trailing
/// space is a miserable thing to debug.
fn parse_key_file(contents: &str) -> Option<String> {
    contents
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            line.trim_start_matches("OPENROUTER_API_KEY=")
                .trim_matches('"')
                .trim_matches('\'')
                .trim()
                .to_string()
        })
        .filter(|key| !key.is_empty())
}

#[derive(Debug, Default)]
pub struct OpenRouterBroker;

impl OpenRouterBroker {
    fn env_key() -> Option<String> {
        std::env::var(API_KEY_VAR)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    }

    fn file_key() -> Option<String> {
        std::fs::read_to_string(key_path())
            .ok()
            .as_deref()
            .and_then(parse_key_file)
    }

    /// Where the active credential came from, for the status line.
    fn source(&self) -> Option<(&'static str, String)> {
        if let Some(key) = Self::env_key() {
            return Some((API_KEY_VAR, key));
        }
        Self::file_key().map(|key| ("~/.patinae/openrouter.key", key))
    }
}

impl AuthBroker for OpenRouterBroker {
    /// OpenRouter authenticates with `Authorization: Bearer <key>`.
    ///
    /// Despite being an API key rather than an OAuth token it goes on the bearer
    /// header, so [`Credential::Bearer`] is the correct shape here — the enum
    /// records how a credential is *presented*, not how it was obtained.
    fn credential(&self) -> Result<Credential, AuthError> {
        Self::env_key()
            .or_else(Self::file_key)
            .map(Credential::Bearer)
            .ok_or_else(|| AuthError::ToolMissing(SETUP_HINT.to_string()))
    }

    fn status(&self) -> AuthStatus {
        match self.source() {
            Some((where_from, _)) => AuthStatus::signed_in(format!("Signed in with {where_from}.")),
            None => AuthStatus::signed_out(SETUP_HINT.to_string()),
        }
    }

    /// There is no interactive flow to run, so this reports what to do instead
    /// of opening a browser the user did not ask for.
    fn login(&self) -> Result<String, AuthError> {
        match self.source() {
            Some((where_from, _)) => Ok(format!(
                "Already using an OpenRouter key from {where_from}; no sign-in needed."
            )),
            None => Err(AuthError::ToolMissing(SETUP_HINT.to_string())),
        }
    }

    /// Remove the key file, which is this provider's credential store.
    ///
    /// Deleting a file the user wrote is a real cost, but a Sign out button that
    /// leaves the agent signed in is worse, and an OpenRouter key is reissued in
    /// a couple of clicks. An environment variable cannot be unset for the
    /// running process's children in any useful way, so it is reported rather
    /// than silently ignored.
    fn logout(&self) -> Result<(), AuthError> {
        let path = key_path();
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| {
                AuthError::Failed(format!("could not remove {}: {e}", path.display()))
            })?;
        }
        if Self::env_key().is_some() {
            return Err(AuthError::Failed(format!(
                "{API_KEY_VAR} is still set in the environment, so OpenRouter is still \
                 signed in. Unset it and restart Patinae to sign out completely."
            )));
        }
        Ok(())
    }

    /// Nothing is cached, so there is nothing to drop. Kept explicit rather than
    /// defaulted so the absence of a cache is a decision on the record.
    fn invalidate(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_key_file_is_read() {
        assert_eq!(
            parse_key_file("sk-or-v1-abc\n").as_deref(),
            Some("sk-or-v1-abc")
        );
    }

    #[test]
    fn the_shapes_people_actually_write_by_hand_all_work() {
        // Every one of these is a silent auth failure if parsed naively.
        for (raw, what) in [
            ("  sk-or-v1-abc  \n", "surrounding whitespace"),
            ("\n\nsk-or-v1-abc\n", "leading blank lines"),
            ("# my openrouter key\nsk-or-v1-abc\n", "a comment header"),
            ("OPENROUTER_API_KEY=sk-or-v1-abc\n", "an export-style line"),
            ("\"sk-or-v1-abc\"\n", "quotes"),
            ("'sk-or-v1-abc'\n", "single quotes"),
            ("sk-or-v1-abc\nsome trailing note\n", "a trailing note"),
        ] {
            assert_eq!(
                parse_key_file(raw).as_deref(),
                Some("sk-or-v1-abc"),
                "failed on {what}"
            );
        }
    }

    #[test]
    fn an_empty_or_comment_only_file_yields_no_key() {
        for raw in ["", "   \n\n", "# nothing here\n", "\"\"\n"] {
            assert!(parse_key_file(raw).is_none(), "{raw:?} should yield no key");
        }
    }

    #[test]
    fn the_key_file_sits_in_the_patinae_config_directory() {
        let path = key_path();
        assert_eq!(path.file_name().unwrap(), KEY_FILE_NAME);
        assert_eq!(
            path.parent().unwrap(),
            patinae_settings::paths::config_dir()
        );
    }

    #[test]
    fn the_hint_names_both_supported_paths_and_where_to_get_a_key() {
        assert!(SETUP_HINT.contains(API_KEY_VAR));
        assert!(SETUP_HINT.contains(KEY_FILE_NAME));
        assert!(SETUP_HINT.contains("openrouter.ai/keys"));
    }

    /// Env vars are process-global; keep the mutation in one test and restore.
    #[test]
    fn an_env_key_is_presented_as_a_bearer_credential() {
        let saved = std::env::var(API_KEY_VAR).ok();
        std::env::set_var(API_KEY_VAR, "  sk-or-test  ");

        assert_eq!(
            OpenRouterBroker.credential().unwrap(),
            Credential::Bearer("sk-or-test".to_string()),
            "OpenRouter takes the key on the bearer header, not as an API-key header"
        );
        assert!(OpenRouterBroker.status().signed_in);

        // A blank value must not count as configured.
        std::env::set_var(API_KEY_VAR, "   ");
        assert!(OpenRouterBroker::env_key().is_none());

        match saved {
            Some(v) => std::env::set_var(API_KEY_VAR, v),
            None => std::env::remove_var(API_KEY_VAR),
        }
    }
}
