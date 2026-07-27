//! Claude assistant core for Patinae.
//!
//! This crate holds the host-agnostic half of the in-app Claude interface:
//! credentials, configuration, the Messages API wire format, stream parsing,
//! tool schemas, command risk classification, and the conversation state
//! machine.
//!
//! It deliberately depends on no rendering, scene, or UI crate so that the
//! agent core stays unit-testable and reusable from the web and Python front
//! ends. Nothing here performs I/O: the host owns the HTTP transport and
//! executes tools against the live viewer.
//!
//! # Shape of a turn
//!
//! [`Agent`] is a state machine the host drives from its frame loop, because
//! executing a tool needs main-thread access to the viewer:
//!
//! 1. [`Agent::submit`] records the user's request and asks for a request.
//! 2. The host streams the response, feeding bytes through
//!    [`stream::SseParser`] into [`stream::MessageAccumulator`].
//! 3. [`Agent::turn_completed`] classifies any tool calls, running the safe
//!    ones and surfacing the rest for approval.
//! 4. [`Agent::tool_finished`] collects results; once the turn is fully
//!    answered the agent asks for the next request.
//!
//! # Safety boundary
//!
//! Claude acts by emitting viewer command strings, so [`approval`] is the
//! security boundary. Unknown commands — including anything a plugin
//! registered at runtime — classify as [`CommandRisk::Destructive`] and always
//! prompt, and a multi-clause string is classified by its riskiest clause so a
//! harmless prefix cannot launder a destructive suffix.
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

pub mod agent;
pub mod approval;
pub mod config;
pub mod prompt;
pub mod schema;
pub mod secret;
pub mod stream;
pub mod tools;
pub mod types;

pub use agent::{Agent, AgentAction, AgentState};
pub use approval::CommandRisk;
pub use config::{ClaudeConfig, Effort, KeySource, API_KEY_ENV, CONFIG_FILE_NAME, DEFAULT_MODEL};
pub use prompt::SceneSummary;
pub use schema::CommandDoc;
pub use secret::{sanitize, SecretString, KEY_PREFIX};
