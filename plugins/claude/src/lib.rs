//! Claude plugin for Patinae.
//!
//! Puts Claude inside the viewer as an agent: the user signs in with their
//! Claude account, asks for something in natural language, and Claude drives the
//! scene directly — issuing Patinae commands, running Python, reading loaded
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
//! There is no official Anthropic Rust SDK, so [`api`] speaks raw HTTPS against
//! `POST /v1/messages` and decodes the SSE stream by hand.

pub mod api;
pub mod auth;
pub mod handler;
pub mod panel;
pub mod prompt;
pub mod settings;
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
    name: "claude",
    description: "Claude AI agent: drives the viewer from natural language",
    commands: [ClaudeCommand],
    panels: [panel::ChatPanel::new(shared_state())],
    settings: [settings::ClaudeSettings],
    register: |reg| {
        // String-valued settings cannot be expressed via
        // `define_plugin_settings!`, so they are registered directly.
        let store = settings::ClaudeSettings::default().init_store();
        reg.register_settings(settings::string_descriptors(), store);
        reg.set_message_handler(handler::ClaudeHandler::new(
            worker::spawn(),
            shared_state(),
        ));
    },
}

/// `claude <prompt>` — hand a natural-language request to the agent.
///
/// Doubles as the point where the system prompt is built: the command registry
/// is reachable from `CommandContext` but not from `PollContext`, so the
/// inventory is captured here on first use.
struct ClaudeCommand;

impl Command for ClaudeCommand {
    fn name(&self) -> &str {
        "claude"
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
                ctx.show_panel("claude_chat");
            }
            Err(_) => ctx.print_error("Claude panel state is unavailable."),
        }
        Ok(())
    }

    command_help! {
        CMD "claude"
        DESCRIPTION ["sends a natural-language request to the Claude agent."]
        REQUIRED [
            { "prompt", "string", "what you want Claude to do" },
        ]
        OPTIONAL []
        EXAMPLES [
            "claude load 1crn and show it as cartoon",
            "claude color the binding site by hydrophobicity",
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
