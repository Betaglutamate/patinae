//! System prompt assembly.
//!
//! Two layers: a static skill document describing Patinae's model and idioms,
//! and a command inventory generated from the *live* command registry so it can
//! never drift from the build (plugin-supplied commands included).
//!
//! The registry is only reachable from a `Command::execute` — `PollContext` has
//! no handle on it — so the prompt is built on the first `claude` invocation and
//! cached for the life of the process.
//!
//! The result must stay byte-identical across turns: it is the cached prefix,
//! and any per-request variation silently costs a full re-read of several
//! thousand tokens.

use std::sync::OnceLock;

/// The static half of the prompt, embedded at build time.
const SKILL: &str = include_str!("../assets/patinae-skill.md");

static PROMPT: OnceLock<String> = OnceLock::new();

/// The assembled prompt, if it has been built.
pub fn cached() -> Option<String> {
    PROMPT.get().cloned()
}

/// Build and cache the prompt from the live command list.
///
/// `commands` is `(name, description)` pairs. Subsequent calls are no-ops, which
/// keeps the cached prefix stable even if commands are registered later.
pub fn init(commands: &[(String, String)]) {
    let _ = PROMPT.set(assemble(commands));
}

fn assemble(commands: &[(String, String)]) -> String {
    let mut out = String::with_capacity(SKILL.len() + commands.len() * 64 + 256);
    out.push_str(SKILL);

    out.push_str("\n\n## Command reference\n\n");
    if commands.is_empty() {
        out.push_str("(command registry unavailable)\n");
        return out;
    }
    out.push_str(
        "Generated from this build's command registry, so it reflects exactly what is \
         available — including commands added by other plugins.\n\n",
    );

    let mut sorted: Vec<&(String, String)> = commands.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, description) in sorted {
        if description.trim().is_empty() {
            out.push_str(&format!("- `{name}`\n"));
        } else {
            out.push_str(&format!("- `{name}` — {}\n", description.trim()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmds() -> Vec<(String, String)> {
        vec![
            ("show".to_string(), "Show a representation".to_string()),
            ("load".to_string(), "Load a structure file".to_string()),
            ("ray".to_string(), String::new()),
        ]
    }

    #[test]
    fn includes_the_skill_document() {
        let p = assemble(&cmds());
        assert!(p.contains("Selection language"));
        assert!(p.contains("byres"));
    }

    /// Just the generated inventory. The skill document also mentions command
    /// names in backticks, so assertions about ordering must not see it.
    fn reference_section(prompt: &str) -> String {
        prompt
            .split_once("## Command reference")
            .expect("command reference section")
            .1
            .to_string()
    }

    #[test]
    fn lists_commands_alphabetically_with_descriptions() {
        let section = reference_section(&assemble(&cmds()));
        let load = section.find("`load`").unwrap();
        let ray = section.find("`ray`").unwrap();
        let show = section.find("`show`").unwrap();
        assert!(load < ray && ray < show, "commands should be sorted");
        assert!(section.contains("`load` — Load a structure file"));
    }

    #[test]
    fn commands_without_descriptions_still_appear() {
        let p = assemble(&cmds());
        assert!(reference_section(&p).contains("- `ray`\n"));
    }

    #[test]
    fn assembly_is_deterministic() {
        // The prompt is the cached prefix; instability costs a full cache miss
        // on every turn.
        assert_eq!(assemble(&cmds()), assemble(&cmds()));
    }

    #[test]
    fn input_order_does_not_change_the_output() {
        let mut reversed = cmds();
        reversed.reverse();
        assert_eq!(assemble(&cmds()), assemble(&reversed));
    }

    #[test]
    fn degrades_gracefully_without_a_registry() {
        let p = assemble(&[]);
        assert!(p.contains("Selection language"));
        assert!(p.contains("command registry unavailable"));
    }

    #[test]
    fn warns_about_the_screenshot_highlight_caveat() {
        // A real behavioural trap: captures exclude selection overlays.
        let p = assemble(&cmds());
        assert!(p.contains("do not include selection"));
    }
}
