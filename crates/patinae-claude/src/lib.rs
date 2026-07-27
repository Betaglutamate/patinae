//! Claude assistant core for Patinae.
//!
//! This crate holds the host-agnostic half of the in-app Claude interface:
//! credential handling, configuration, and (in later steps) the Messages API
//! client, tool schema, and agent state machine.
//!
//! It deliberately depends on no rendering, scene, or UI crate so that the
//! agent core stays unit-testable and reusable from the web and Python front
//! ends.
//!
//! # Credential handling
//!
//! Patinae persists and shares a lot of state — settings, `.prs` sessions,
//! command history, the output log — so an API key that reaches the wrong
//! place can end up in a user's git repository. Two rules keep that from
//! happening, and both are enforced by tests rather than convention:
//!
//! - The key is held in a [`SecretString`], which cannot be printed, logged,
//!   or `Debug`-formatted, and wipes its backing bytes on drop.
//! - The key is stored only in `<config_dir>/claude.json` (owner-only, outside
//!   any repository) or read from `ANTHROPIC_API_KEY`. It is never a setting,
//!   never a command argument, and never part of a session file.
//!
//! [`SecretString::expose`] is the only accessor, so grepping for it audits
//! every path by which the key can escape.

pub mod config;
pub mod secret;

pub use config::{ClaudeConfig, Effort, KeySource, API_KEY_ENV, CONFIG_FILE_NAME, DEFAULT_MODEL};
pub use secret::{sanitize, SecretString, KEY_PREFIX};
