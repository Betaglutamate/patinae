//! AI agent plugin for Patinae.
//!
//! Puts an AI agent inside the viewer: the user signs in with their Claude or
//! Google account, asks for something in natural language, and the agent drives
//! the scene directly — issuing Patinae commands, running Python, reading loaded
//! structures and sequence data, and looking at screenshots to verify its work.
//!
//! # Architecture
//!
//! Two threads, connected by channels:
//!
//! - A **worker thread** ([`worker`]) owns a Tokio runtime and talks to the
//!   Messages API. It never touches viewer state.
//! - The **main thread** ([`handler`]) drains the worker in
//!   `MessageHandler::poll` and applies every tool call through `PollContext`,
//!   which is the only sanctioned way for a plugin to reach the viewer.
//!
//! Neither vendor ships a Rust SDK, so [`provider`] speaks raw HTTPS and decodes
//! each vendor's SSE stream by hand behind a common trait.

pub mod auth;
pub mod handler;
pub mod panel;
pub mod prompt;
pub mod provider;
pub mod settings;
pub mod sse;
pub mod state;
pub mod tools;
pub mod worker;

use std::sync::OnceLock;

use patinae_plugin::patinae_plugin;
use patinae_plugin::prelude::*;

use crate::state::Shared;

/// Shared panel/handler state.
///
/// The `patinae_plugin!` macro constructs panels and the message handler in
/// separate expressions, so the two halves meet through this rather than being
/// threaded from a common constructor.
fn shared_state() -> Shared {
    static STATE: OnceLock<Shared> = OnceLock::new();
    STATE.get_or_init(state::new_shared).clone()
}

patinae_plugin! {
    name: "ai",
    description: "AI agent (Claude or Gemini): drives the viewer from natural language",
    commands: [AiCommand],
    panels: [panel::ChatPanel::new(shared_state())],
    settings: [settings::AiSettings],
    register: |reg| {
        // String-valued settings cannot be expressed via
        // `define_plugin_settings!`, so they are registered directly.
        let store = settings::AiSettings::default().init_store();
        reg.register_settings(settings::string_descriptors(), store);
        reg.set_message_handler(handler::ClaudeHandler::new(
            worker::spawn(),
            shared_state(),
        ));
    },
}

/// `ai <prompt>` — hand a natural-language request to the agent.
///
/// Doubles as the point where the system prompt is built: the command registry
/// is reachable from `CommandContext` but not from `PollContext`, so the
/// inventory is captured here on first use.
struct AiCommand;

impl Command for AiCommand {
    fn name(&self) -> &str {
        "ai"
    }

    /// Provider names are accepted as aliases purely for discoverability —
    /// they do not switch provider. Use `set ai_provider` for that.
    fn aliases(&self) -> &[&str] {
        &["claude", "gemini"]
    }

    fn execute<'v, 'r>(
        &self,
        ctx: &mut CommandContext<'v, 'r, dyn ViewerLike + 'v>,
        args: &ParsedCommand,
    ) -> CmdResult {
        ensure_prompt(ctx);

        let prompt = args.get_str(0).unwrap_or("").trim().to_string();
        if prompt.is_empty() {
            ctx.print_warning("usage: claude <prompt>");
            return Ok(());
        }

        match shared_state().lock() {
            Ok(mut state) => {
                state.submit = Some(prompt);
                state.dirty = true;
                ctx.show_panel("ai_chat");
            }
            Err(_) => ctx.print_error("AI panel state is unavailable."),
        }
        Ok(())
    }

    command_help! {
        CMD "ai"
        DESCRIPTION ["sends a natural-language request to the AI agent."]
        REQUIRED [
            { "prompt", "string", "what you want Claude to do" },
        ]
        OPTIONAL []
        EXAMPLES [
            "ai load 1crn and show it as cartoon",
            "ai color the binding site by hydrophobicity",
        ]
    }

    fn arg_hints(&self) -> &[ArgHint] {
        &[ArgHint::None]
    }

    fn runtime_requirements(&self) -> CommandRuntimeRequirements {
        // Only reads args and hands off to the worker.
        CommandRuntimeRequirements::NONE
    }
}

/// Build the system prompt from the live command registry, once.
fn ensure_prompt<'v, 'r>(ctx: &CommandContext<'v, 'r, dyn ViewerLike + 'v>) {
    if prompt::cached().is_some() {
        return;
    }
    let Some(registry) = ctx.registry() else {
        return;
    };
    let entries: Vec<(String, String)> = registry
        .names()
        .map(|name| {
            let description = registry
                .get(name)
                .map(|c| c.description().to_string())
                .unwrap_or_default();
            (name.to_string(), description)
        })
        .collect();
    prompt::init(&entries);
}
