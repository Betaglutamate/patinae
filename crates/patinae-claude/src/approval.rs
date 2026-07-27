//! Risk classification for viewer commands.
//!
//! Claude drives Patinae by emitting command strings, so this module is the
//! security boundary of the whole feature. Two rules shape it:
//!
//! - **Unknown commands are [`CommandRisk::Destructive`].** Plugins can
//!   register arbitrary commands at runtime, and this crate cannot know what
//!   they do. Defaulting to "ask" is the only safe answer.
//! - **A command string is classified by its riskiest clause.** `zoom` is
//!   harmless and `delete all` is not, but `zoom; delete all` is a single
//!   string, and classifying it by the first clause would wave the delete
//!   straight through.

/// How dangerous a command is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CommandRisk {
    /// Reads state without changing anything.
    ReadOnly,
    /// Changes what is displayed, reversibly.
    Mutating,
    /// Can lose work, touch the filesystem, or exit the app.
    Destructive,
}

impl CommandRisk {
    /// Returns whether this command needs an explicit click before running.
    ///
    /// `auto_approve_mutating` reflects the user's panel setting; destructive
    /// commands always ask regardless.
    pub fn requires_approval(self, auto_approve_mutating: bool) -> bool {
        match self {
            Self::ReadOnly => false,
            Self::Mutating => !auto_approve_mutating,
            Self::Destructive => true,
        }
    }

    /// Short label for the approval UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::Mutating => "changes the view",
            Self::Destructive => "destructive",
        }
    }
}

/// Commands that only read state.
pub const READ_ONLY: &[&str] = &[
    "get",
    "get_view",
    "count_states",
    "ls",
    "dir",
    "pwd",
    "help",
];

/// Commands that change the scene reversibly.
///
/// Includes the aliases Patinae registers, because classification happens
/// before any alias resolution.
pub const MUTATING: &[&str] = &[
    "show",
    "hide",
    "show_as",
    "as",
    "color",
    "colour",
    "bg_color",
    "bg_colour",
    "background",
    "set_color",
    "set_colour",
    "select",
    "deselect",
    "indicate",
    "zoom",
    "center",
    "orient",
    "reset",
    "clip",
    "turn",
    "move",
    "origin",
    "viewport",
    "full_screen",
    "fullscreen",
    "set",
    "unset",
    "label",
    "enable",
    "disable",
    "toggle",
    "scene",
    "view",
    "dss",
    "distance",
    "dist",
    "angle",
    "dihedral",
    "rock",
    "frame",
    "forward",
    "backward",
    "rewind",
    "middle",
    "ending",
    "mplay",
    "mstop",
    "mpause",
    "mtoggle",
    "refresh",
    "rebuild",
    "translate",
    "rotate",
    "align",
    "rmsd",
    "rms_cur",
    "isomesh",
    "isodot",
    "isosurface",
    "symexp",
    "group",
    "ungroup",
    "state",
];

/// Commands that can lose work, write files, or exit.
///
/// Classification also reaches this outcome via the "unknown means
/// destructive" default, so this list is not load-bearing for safety. It is
/// public so the panel can explain which commands always prompt.
pub const DESTRUCTIVE: &[&str] = &[
    "delete",
    "remove",
    "reinitialize",
    "reinit",
    "clear",
    "quit",
    "exit",
    "save",
    "png",
    "load",
    "load_traj",
    "fetch",
    "run",
    "@",
    "/",
    "python",
    "cd",
    "create",
    "copy",
    "extract",
    "set_name",
    "rename",
    "split_states",
    "delete_states",
    "delete_state",
    "alter",
    "iterate",
    "mset",
    "madd",
    "mview",
    "mpng",
    "mproduce",
    "setup_ccd",
    "transform_selection",
];

/// Classifies a whole command string, which may contain several clauses.
///
/// Returns the riskiest classification across all clauses. An empty or
/// comment-only string is [`CommandRisk::ReadOnly`] because it executes
/// nothing.
pub fn classify(command: &str) -> CommandRisk {
    let clauses = split_clauses(command);
    if clauses.is_empty() {
        return CommandRisk::ReadOnly;
    }
    clauses
        .iter()
        .map(|clause| classify_clause(clause))
        .max()
        .unwrap_or(CommandRisk::Destructive)
}

/// Classifies a single command clause by its leading command name.
fn classify_clause(clause: &str) -> CommandRisk {
    let Some(name) = command_name(clause) else {
        // Nothing to run.
        return CommandRisk::ReadOnly;
    };
    classify_name(&name)
}

/// Classifies a bare command name.
pub fn classify_name(name: &str) -> CommandRisk {
    let name = name.trim().to_ascii_lowercase();
    if name.is_empty() {
        return CommandRisk::ReadOnly;
    }
    if READ_ONLY.contains(&name.as_str()) {
        return CommandRisk::ReadOnly;
    }
    if MUTATING.contains(&name.as_str()) {
        return CommandRisk::Mutating;
    }
    // Everything else, including every plugin-registered command, asks first.
    CommandRisk::Destructive
}

/// Extracts the command name from one clause.
///
/// Patinae accepts `@script.pml` and `/expr` as prefix aliases with no
/// separating space, so those are recognised before generic tokenising.
fn command_name(clause: &str) -> Option<String> {
    let trimmed = clause.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    if let Some(first) = trimmed.chars().next() {
        if first == '@' || first == '/' {
            return Some(first.to_string());
        }
    }
    let end = trimmed
        .find(|c: char| c.is_whitespace() || c == ',')
        .unwrap_or(trimmed.len());
    Some(trimmed[..end].to_string())
}

/// Splits a command string on `;`, ignoring separators inside quotes.
///
/// Quote handling matters: `select x, resn "A;B"` is one command, and a naive
/// split would classify the fragment after the semicolon as an unknown — which
/// is merely noisy — but the same bug in reverse would let a real second
/// command hide inside what looks like a quoted string.
fn split_clauses(command: &str) -> Vec<String> {
    let mut clauses = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for ch in command.chars() {
        match ch {
            '"' | '\'' => {
                match quote {
                    Some(open) if open == ch => quote = None,
                    Some(_) => {}
                    None => quote = Some(ch),
                }
                current.push(ch);
            }
            ';' if quote.is_none() => {
                if !current.trim().is_empty() {
                    clauses.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        clauses.push(current.trim().to_string());
    }
    clauses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_commands_are_recognised() {
        for command in ["get sphere_scale", "ls", "pwd", "help zoom", "get_view"] {
            assert_eq!(
                classify(command),
                CommandRisk::ReadOnly,
                "{command} should be read-only"
            );
        }
    }

    #[test]
    fn scene_changes_are_mutating() {
        for command in [
            "show cartoon",
            "color red, chain A",
            "zoom",
            "set sphere_scale, 0.4",
            "select pocket, byres around 5 ligand",
        ] {
            assert_eq!(
                classify(command),
                CommandRisk::Mutating,
                "{command} should be mutating"
            );
        }
    }

    #[test]
    fn destructive_commands_are_recognised() {
        for command in [
            "delete all",
            "quit",
            "save /tmp/out.pdb",
            "load /tmp/in.pdb",
            "fetch 1ubq",
            "reinitialize",
            "run script.pml",
            "png /tmp/shot.png",
        ] {
            assert_eq!(
                classify(command),
                CommandRisk::Destructive,
                "{command} should be destructive"
            );
        }
    }

    #[test]
    fn every_listed_destructive_command_classifies_as_destructive() {
        for name in DESTRUCTIVE {
            assert_eq!(
                classify_name(name),
                CommandRisk::Destructive,
                "{name} should be destructive"
            );
        }
    }

    #[test]
    fn unknown_and_plugin_commands_default_to_destructive() {
        // A plugin can register anything at runtime; this crate cannot know
        // what it does, so it must ask.
        for command in ["ray 800, 600", "some_future_command", "hello Alice"] {
            assert_eq!(
                classify(command),
                CommandRisk::Destructive,
                "{command} should default to destructive"
            );
        }
    }

    #[test]
    fn a_multi_clause_string_takes_the_riskiest_clause() {
        // The important case: a harmless-looking prefix must not launder a
        // destructive suffix past the approval gate.
        assert_eq!(classify("zoom; delete all"), CommandRisk::Destructive);
        assert_eq!(classify("delete all; zoom"), CommandRisk::Destructive);
        assert_eq!(classify("show cartoon; zoom"), CommandRisk::Mutating);
        assert_eq!(classify("get x; ls"), CommandRisk::ReadOnly);
    }

    #[test]
    fn semicolons_inside_quotes_do_not_split_clauses() {
        // One command, not two. Were this split, the tail would be classified
        // as an unknown command and needlessly prompt.
        assert_eq!(
            classify(r#"select weird, resn "A;B""#),
            CommandRisk::Mutating
        );
        assert_eq!(
            split_clauses(r#"select weird, resn "A;B""#).len(),
            1,
            "quoted semicolon should not split"
        );
    }

    #[test]
    fn script_and_python_prefix_aliases_are_destructive() {
        assert_eq!(classify("@setup.pml"), CommandRisk::Destructive);
        assert_eq!(classify("/print(1)"), CommandRisk::Destructive);
    }

    #[test]
    fn classification_ignores_case_and_surrounding_whitespace() {
        assert_eq!(classify("  ZOOM  "), CommandRisk::Mutating);
        assert_eq!(classify("Delete all"), CommandRisk::Destructive);
    }

    #[test]
    fn comments_and_empty_input_execute_nothing() {
        assert_eq!(classify(""), CommandRisk::ReadOnly);
        assert_eq!(classify("   "), CommandRisk::ReadOnly);
        assert_eq!(classify("# just a note"), CommandRisk::ReadOnly);
    }

    #[test]
    fn command_names_split_on_commas_as_well_as_spaces() {
        // `show,cartoon` is valid input; the name must not become "show,cartoon".
        assert_eq!(classify("show,cartoon"), CommandRisk::Mutating);
    }

    #[test]
    fn approval_policy_respects_the_auto_approve_setting() {
        assert!(!CommandRisk::ReadOnly.requires_approval(false));
        assert!(!CommandRisk::ReadOnly.requires_approval(true));

        assert!(CommandRisk::Mutating.requires_approval(false));
        assert!(!CommandRisk::Mutating.requires_approval(true));

        // Destructive always asks, whatever the setting says.
        assert!(CommandRisk::Destructive.requires_approval(true));
        assert!(CommandRisk::Destructive.requires_approval(false));
    }

    #[test]
    fn risk_ordering_puts_destructive_highest() {
        assert!(CommandRisk::Destructive > CommandRisk::Mutating);
        assert!(CommandRisk::Mutating > CommandRisk::ReadOnly);
    }
}
