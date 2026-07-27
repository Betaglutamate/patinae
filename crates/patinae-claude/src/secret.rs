//! Credential handling primitives.
//!
//! Patinae persists and shares a lot of state — settings, `.prs` sessions,
//! command history, the output log. An API key that reaches any of those can
//! end up in a user's git repository. This module provides the two primitives
//! that keep that from happening:
//!
//! - [`SecretString`], which cannot be printed, logged, or `Debug`-formatted,
//!   and whose backing bytes are wiped on drop.
//! - [`sanitize`], a last-resort scrubber applied to every string that reaches
//!   the UI or the log.
//!
//! The single accessor [`SecretString::expose`] is deliberately the only way to
//! read the value back, so `rg 'expose\(\)'` is a complete audit of where the
//! key can escape.

use std::fmt;
use std::sync::atomic::{compiler_fence, Ordering};

/// Prefix shared by Anthropic API keys.
pub const KEY_PREFIX: &str = "sk-ant-";

/// Placeholder rendered in place of a secret value.
const REDACTED: &str = "<redacted>";

/// Number of trailing characters revealed by [`SecretString::fingerprint`].
const FINGERPRINT_TAIL: usize = 4;

/// A string that never reveals itself through `Debug`, `Display`, or logging.
///
/// The backing bytes are overwritten when the value is dropped. This does not
/// defend against an attacker with code execution or memory forensics — it
/// narrows the window in which a released allocation still holds the key.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    /// Wraps a value as a secret.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the underlying value.
    ///
    /// This is the only way to read a secret back. It should be called in
    /// exactly one place — building the outbound `x-api-key` header — so that
    /// grepping for it audits every escape route.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Returns `true` when the secret holds no characters.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns a display-safe identifier such as `sk-ant-…a1b2`.
    ///
    /// Reveals at most the key prefix and the last [`FINGERPRINT_TAIL`]
    /// characters, never the middle. Short values are fully masked so a
    /// truncated or malformed key cannot be reconstructed from its
    /// fingerprint.
    pub fn fingerprint(&self) -> String {
        let prefix = if self.0.starts_with(KEY_PREFIX) {
            KEY_PREFIX
        } else {
            ""
        };
        let tail_start = self.0.len().saturating_sub(FINGERPRINT_TAIL);
        if tail_start <= prefix.len() {
            return format!("{prefix}…");
        }
        // Only slice on a character boundary; API keys are ASCII, but a
        // pasted value may not be.
        let tail = self
            .0
            .char_indices()
            .find(|(idx, _)| *idx >= tail_start)
            .map_or("", |(idx, _)| &self.0[idx..]);
        format!("{prefix}…{tail}")
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretString({REDACTED})")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        // SAFETY: every byte is overwritten with 0x00, which is valid UTF-8
        // (NUL is a single-byte code point), so the `String` invariant holds
        // for the remainder of the drop. `write_volatile` keeps the compiler
        // from eliding the writes as dead stores.
        unsafe {
            for byte in self.0.as_bytes_mut() {
                std::ptr::write_volatile(byte, 0);
            }
        }
        compiler_fence(Ordering::SeqCst);
    }
}

/// Removes anything that looks like an Anthropic API key from `text`.
///
/// Applied to every string that reaches the output log, the notification
/// toast, or `log::error!`. The key should never be in those strings to begin
/// with; this is defence in depth against a future change that puts it
/// somewhere unexpected.
pub fn sanitize(text: &str) -> String {
    if !text.contains(KEY_PREFIX) {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(KEY_PREFIX) {
        out.push_str(&rest[..at]);
        out.push_str(REDACTED);
        let after = &rest[at + KEY_PREFIX.len()..];
        let end = after.find(|c: char| !is_key_char(c)).unwrap_or(after.len());
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

/// Characters that may appear in the body of an API key.
fn is_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-' || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Realistic-looking but fake credential. The literal is split so the CI
    /// secret scan (which matches `sk-ant-` followed by 20+ key characters)
    /// does not flag this file; the constant still has full key shape at
    /// runtime.
    const FAKE_KEY: &str = concat!("sk-ant-", "test-DEADBEEF0123456789");

    #[test]
    fn debug_and_display_do_not_reveal_the_secret() {
        let secret = SecretString::new(FAKE_KEY);

        let debug = format!("{secret:?}");
        let display = format!("{secret}");

        assert!(!debug.contains(FAKE_KEY), "Debug leaked the key: {debug}");
        assert!(
            !display.contains(FAKE_KEY),
            "Display leaked the key: {display}"
        );
        assert!(!debug.contains("DEADBEEF"));
        assert!(!display.contains("DEADBEEF"));
    }

    #[test]
    fn debug_of_a_containing_struct_does_not_reveal_the_secret() {
        #[derive(Debug)]
        struct Holder {
            key: SecretString,
            model: &'static str,
        }

        let holder = Holder {
            key: SecretString::new(FAKE_KEY),
            model: "claude-opus-5",
        };

        let rendered = format!("{holder:?}");

        assert!(!rendered.contains(FAKE_KEY));
        assert!(rendered.contains("claude-opus-5"));
        // The secret is still readable through the explicit accessor; only the
        // formatting paths are blocked.
        assert_eq!(holder.key.expose(), FAKE_KEY);
        assert_eq!(holder.model, "claude-opus-5");
    }

    #[test]
    fn expose_returns_the_value() {
        assert_eq!(SecretString::new(FAKE_KEY).expose(), FAKE_KEY);
    }

    #[test]
    fn fingerprint_reveals_only_prefix_and_tail() {
        let fingerprint = SecretString::new(FAKE_KEY).fingerprint();

        assert_eq!(fingerprint, "sk-ant-…6789");
        assert!(!fingerprint.contains("DEADBEEF"));
        assert!(fingerprint.len() < FAKE_KEY.len());
    }

    #[test]
    fn fingerprint_masks_short_values_entirely() {
        assert_eq!(SecretString::new("sk-ant-").fingerprint(), "sk-ant-…");
        assert_eq!(SecretString::new("sk-ant-ab").fingerprint(), "sk-ant-…");
        assert_eq!(SecretString::new("").fingerprint(), "…");
    }

    #[test]
    fn fingerprint_handles_non_ascii_without_panicking() {
        // A pasted value may be anything; slicing must stay on char boundaries.
        let fingerprint = SecretString::new("sk-ant-\u{1f600}\u{1f600}").fingerprint();
        assert!(fingerprint.starts_with("sk-ant-…"));
    }

    #[test]
    fn sanitize_strips_keys_from_error_text() {
        let message = format!("request failed: header x-api-key={FAKE_KEY} rejected");

        let cleaned = sanitize(&message);

        assert!(
            !cleaned.contains(FAKE_KEY),
            "sanitize left the key in place"
        );
        assert!(!cleaned.contains("DEADBEEF"));
        assert!(cleaned.contains("request failed"));
        assert!(cleaned.contains("rejected"));
    }

    #[test]
    fn sanitize_strips_every_occurrence() {
        let message = format!("{FAKE_KEY} and also {FAKE_KEY}.");

        let cleaned = sanitize(&message);

        assert!(!cleaned.contains("sk-ant-"));
        assert!(cleaned.ends_with('.'));
        assert_eq!(cleaned.matches(REDACTED).count(), 2);
    }

    #[test]
    fn sanitize_leaves_unrelated_text_untouched() {
        let message = "no credentials here";
        assert_eq!(sanitize(message), message);
    }
}
