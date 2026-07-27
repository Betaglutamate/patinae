//! System prompt assembly.
//!
//! Two things here are deliberate and easy to undo by accident:
//!
//! - **The scene summary is not in the system prompt.** It changes every turn,
//!   and the system prompt is the cached prefix. Putting volatile state there
//!   would invalidate the cache on every request and pay full price forever.
//!   It goes in the user turn instead, after the last breakpoint.
//! - **There is no "verify your work" instruction.** Current models self-verify;
//!   telling them to verify produces over-verification. Its absence is a
//!   decision, not an oversight, and a test guards it.

use crate::types::ContentBlock;

/// Role and behavioural guidance. Stable across turns, so it caches.
const PREAMBLE: &str = "\
You are Claude, embedded in Patinae, a GPU-accelerated molecular viewer. You help \
the user inspect and present molecular structures by running viewer commands on \
their behalf.

The command line is the viewer's universal interface: almost anything the UI can \
do, a command can do. Use the `run_command` tool to act, and the inspection tools \
to find out what is actually in the scene rather than assuming.

Working style:

- Inspect before you act when the scene state matters. `list_objects` and \
`count_atoms` are cheap; a command that silently selects zero atoms is not.
- Prefer one command per tool call. When something fails you then know exactly \
which step broke.
- If a command errors, read the message and adapt. Do not repeat the same command \
unchanged.
- Some commands need the user's approval before they run. If one is denied, \
respect it: acknowledge and offer an alternative rather than trying to route \
around the refusal.

Communicating:

- Keep responses focused and brief. Lead with the outcome — what changed, or what \
you found — then any detail that would change what the user does next.
- Answer in prose. Reach for structure only when the content is genuinely a list \
or a table.
- Molecular jargon is fine; the user is a structural biologist. Invented shorthand \
is not — spell out object and selection names in full.

Scope:

- Deliver the change the user asked for, at the scope they intended. Make routine \
judgement calls yourself; check in only when different readings would lead to \
materially different work.
- Do not restyle the scene beyond the request. If you think a different approach \
is better, say so in a sentence and do what was asked.";

/// Builds the system prompt blocks for a request.
///
/// The final block carries the cache breakpoint, so the preamble, the command
/// reference, and every tool schema ahead of it are cached together.
pub fn system_blocks(command_reference: &str) -> Vec<ContentBlock> {
    let mut text = String::with_capacity(PREAMBLE.len() + command_reference.len() + 2);
    text.push_str(PREAMBLE);
    if !command_reference.trim().is_empty() {
        text.push_str("\n\n");
        text.push_str(command_reference.trim_end());
    }
    // A single cached block: fewer breakpoints, and nothing volatile can slip
    // between the preamble and the reference.
    vec![ContentBlock::cached_text(text)]
}

/// A snapshot of the scene, rendered into the user turn.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SceneSummary {
    /// Loaded object names.
    pub objects: Vec<String>,
    /// Named selection names.
    pub selections: Vec<String>,
    /// Total atoms currently loaded.
    pub total_atoms: usize,
}

impl SceneSummary {
    /// Renders the summary as prompt text.
    pub fn render(&self) -> String {
        let mut out = String::from("Current scene:\n");
        if self.objects.is_empty() {
            out.push_str("- No objects loaded.\n");
        } else {
            out.push_str("- Objects: ");
            out.push_str(&self.objects.join(", "));
            out.push('\n');
            out.push_str(&format!("- Total atoms: {}\n", self.total_atoms));
        }
        if self.selections.is_empty() {
            out.push_str("- No named selections.\n");
        } else {
            out.push_str("- Named selections: ");
            out.push_str(&self.selections.join(", "));
            out.push('\n');
        }
        out
    }
}

/// Builds the user turn: scene state first, then the request.
///
/// Both live after the cache breakpoint, so the volatile scene summary costs
/// nothing in cache terms.
pub fn user_turn_blocks(summary: &SceneSummary, request: &str) -> Vec<ContentBlock> {
    vec![
        ContentBlock::text(summary.render()),
        ContentBlock::text(request.to_string()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> String {
        "# Patinae command reference\n\n## zoom\nZoom to a selection.\n".to_string()
    }

    #[test]
    fn system_prompt_embeds_the_command_reference() {
        let blocks = system_blocks(&reference());
        let text = blocks[0].as_text().expect("system block should be text");

        assert!(text.contains("Patinae command reference"));
        assert!(text.contains("## zoom"));
        assert!(text.contains("You are Claude"));
    }

    #[test]
    fn the_last_system_block_carries_the_cache_breakpoint() {
        let blocks = system_blocks(&reference());

        let ContentBlock::Text { cache_control, .. } =
            blocks.last().expect("there should be a system block")
        else {
            panic!("system blocks should be text");
        };
        assert!(
            cache_control.is_some(),
            "the cached prefix must end with a breakpoint"
        );
    }

    #[test]
    fn the_system_prompt_holds_nothing_volatile() {
        // The cache-correctness invariant: anything that changes per turn must
        // live in the user turn, not here. If a future change moves the scene
        // summary into the system prompt, prompt caching silently stops working
        // and every turn pays full price.
        let summary = SceneSummary {
            objects: vec!["1ubq".to_string()],
            selections: vec!["pocket".to_string()],
            total_atoms: 602,
        };
        let text = system_blocks(&reference())[0]
            .as_text()
            .expect("system block should be text")
            .to_string();

        assert!(!text.contains("1ubq"));
        assert!(!text.contains("pocket"));
        assert!(!text.contains("602"));
        assert!(!text.contains(&summary.render()));
    }

    #[test]
    fn the_system_prompt_is_byte_stable_across_calls() {
        assert_eq!(system_blocks(&reference()), system_blocks(&reference()));
    }

    #[test]
    fn the_system_prompt_does_not_ask_the_model_to_verify_itself() {
        // Deliberate omission: current models self-verify, and an explicit
        // instruction produces over-verification. Guarded so it is not
        // "helpfully" reinstated.
        let text = system_blocks(&reference())[0]
            .as_text()
            .expect("system block should be text")
            .to_lowercase();

        for phrase in [
            "double-check",
            "double check",
            "verify your work",
            "verification step",
            "re-verify",
        ] {
            assert!(
                !text.contains(phrase),
                "system prompt should not contain {phrase:?}"
            );
        }
    }

    #[test]
    fn the_system_prompt_asks_for_brevity_and_scope_discipline() {
        let text = system_blocks(&reference())[0]
            .as_text()
            .expect("system block should be text")
            .to_lowercase();

        assert!(text.contains("brief"));
        assert!(text.contains("scope"));
    }

    #[test]
    fn an_empty_reference_leaves_no_trailing_separator() {
        let text = system_blocks("")[0]
            .as_text()
            .expect("system block should be text")
            .to_string();

        assert!(text.ends_with("do what was asked."), "got: {text:?}");
    }

    #[test]
    fn scene_summary_renders_objects_selections_and_counts() {
        let summary = SceneSummary {
            objects: vec!["1ubq".to_string(), "ligand".to_string()],
            selections: vec!["pocket".to_string()],
            total_atoms: 602,
        };

        let text = summary.render();

        assert!(text.contains("Objects: 1ubq, ligand"));
        assert!(text.contains("Total atoms: 602"));
        assert!(text.contains("Named selections: pocket"));
    }

    #[test]
    fn an_empty_scene_says_so_explicitly() {
        // "No objects loaded" beats an empty list the model has to interpret.
        let text = SceneSummary::default().render();

        assert!(text.contains("No objects loaded"));
        assert!(text.contains("No named selections"));
        assert!(!text.contains("Total atoms"));
    }

    #[test]
    fn the_user_turn_carries_scene_state_and_the_request() {
        let blocks = user_turn_blocks(&SceneSummary::default(), "show it as cartoon");

        assert_eq!(blocks.len(), 2);
        assert!(blocks[0]
            .as_text()
            .expect("first block is text")
            .contains("Current scene"));
        assert_eq!(
            blocks[1].as_text(),
            Some("show it as cartoon"),
            "the request should be the last block"
        );
    }

    #[test]
    fn user_turn_blocks_are_not_cached() {
        // These change every turn; a breakpoint here would be wasted.
        for block in user_turn_blocks(&SceneSummary::default(), "hi") {
            let ContentBlock::Text { cache_control, .. } = block else {
                panic!("user turn should be text blocks");
            };
            assert!(cache_control.is_none());
        }
    }
}
