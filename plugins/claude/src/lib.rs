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
//! - A **worker thread** owns a Tokio runtime and talks to the Messages API. It
//!   never touches viewer state.
//! - The **main thread** drains the worker in [`MessageHandler::poll`] and
//!   applies every tool call through `PollContext`, which is the only sanctioned
//!   way for a plugin to reach the viewer.
//!
//! There is no official Anthropic Rust SDK, so [`api`] speaks raw HTTPS against
//! `POST /v1/messages` and decodes the SSE stream by hand.

pub mod api;
pub mod auth;

use patinae_plugin::patinae_plugin;
use patinae_plugin::prelude::*;

patinae_plugin! {
    name: "claude",
    description: "Claude AI agent: drives the viewer from natural language",
    commands: [ClaudeCommand],
}

/// `claude <prompt>` — hand a natural-language request to the agent.
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
        let prompt = args.get_str(0).unwrap_or("").trim().to_string();
        if prompt.is_empty() {
            ctx.print_warning("usage: claude <prompt>");
            return Ok(());
        }
        ctx.print(&format!("[claude] {prompt}"));
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
}
