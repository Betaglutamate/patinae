//! Command documentation and tool schema generation.
//!
//! Patinae's commands already carry structured help via the `command_help!`
//! macro, so the model's view of the command surface is generated rather than
//! hand-maintained. The host walks `CommandRegistry` into [`CommandDoc`]s —
//! which picks up plugin-registered commands for free — and this module turns
//! those into the reference text embedded in the system prompt.
//!
//! **Output must be byte-stable.** The reference sits inside the cached prompt
//! prefix, so a reordering between runs would silently disable prompt caching
//! and pay full price on every turn. Everything here sorts deterministically.

use serde::{Deserialize, Serialize};

/// Structured help for one viewer command.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CommandDoc {
    /// Canonical command name.
    pub name: String,
    /// Alternate names accepted by the parser.
    pub aliases: Vec<String>,
    /// One-line summary.
    pub description: String,
    /// Usage line, e.g. `show <representation>, <selection>`.
    pub usage: String,
    /// Argument documentation.
    pub arguments: String,
    /// Example invocations.
    pub examples: Vec<String>,
}

impl CommandDoc {
    /// Builds a doc with only a name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            ..Default::default()
        }
    }
}

/// Renders the command reference embedded in the system prompt.
///
/// Sorting is applied to both commands and aliases so the same registry always
/// produces identical bytes, whatever order the host enumerated it in.
pub fn command_reference(docs: &[CommandDoc]) -> String {
    let mut sorted: Vec<&CommandDoc> = docs.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    sorted.dedup_by(|a, b| a.name == b.name);

    let mut out = String::from("# Patinae command reference\n\n");
    for doc in sorted {
        out.push_str("## ");
        out.push_str(&doc.name);
        if !doc.aliases.is_empty() {
            let mut aliases = doc.aliases.clone();
            aliases.sort();
            aliases.dedup();
            out.push_str(" (aliases: ");
            out.push_str(&aliases.join(", "));
            out.push(')');
        }
        out.push('\n');

        if !doc.description.is_empty() {
            out.push_str(doc.description.trim());
            out.push('\n');
        }
        if !doc.usage.is_empty() {
            out.push_str("Usage: ");
            out.push_str(doc.usage.trim());
            out.push('\n');
        }
        if !doc.arguments.is_empty() {
            out.push_str("Arguments: ");
            out.push_str(doc.arguments.trim());
            out.push('\n');
        }
        for example in &doc.examples {
            if example.trim().is_empty() {
                continue;
            }
            out.push_str("Example: ");
            out.push_str(example.trim());
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

/// Returns the names of every documented command, sorted.
pub fn command_names(docs: &[CommandDoc]) -> Vec<String> {
    let mut names: Vec<String> = docs
        .iter()
        .flat_map(|doc| std::iter::once(doc.name.clone()).chain(doc.aliases.iter().cloned()))
        .collect();
    names.sort();
    names.dedup();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<CommandDoc> {
        vec![
            CommandDoc {
                name: "show".to_string(),
                aliases: vec![],
                description: "Show a representation for a selection.".to_string(),
                usage: "show <representation>, <selection>".to_string(),
                arguments: "representation: cartoon|sticks|spheres".to_string(),
                examples: vec!["show cartoon, chain A".to_string()],
            },
            CommandDoc {
                name: "color".to_string(),
                aliases: vec!["colour".to_string()],
                description: "Color a selection.".to_string(),
                usage: "color <color>, <selection>".to_string(),
                arguments: String::new(),
                examples: vec![],
            },
        ]
    }

    #[test]
    fn reference_lists_commands_alphabetically() {
        let text = command_reference(&sample());

        let color_at = text.find("## color").expect("color should be listed");
        let show_at = text.find("## show").expect("show should be listed");
        assert!(color_at < show_at, "commands should be alphabetical");
    }

    #[test]
    fn reference_includes_aliases_usage_arguments_and_examples() {
        let text = command_reference(&sample());

        assert!(text.contains("## color (aliases: colour)"));
        assert!(text.contains("Usage: show <representation>, <selection>"));
        assert!(text.contains("Arguments: representation: cartoon|sticks|spheres"));
        assert!(text.contains("Example: show cartoon, chain A"));
    }

    #[test]
    fn reference_is_byte_stable_regardless_of_input_order() {
        // The reference lives inside the cached prompt prefix: a single byte of
        // reordering between runs silently disables prompt caching.
        let forward = sample();
        let mut reversed = sample();
        reversed.reverse();

        assert_eq!(command_reference(&forward), command_reference(&reversed));
    }

    #[test]
    fn alias_order_does_not_affect_output() {
        let mut a = CommandDoc::new("bg_color", "Set background.");
        a.aliases = vec!["background".to_string(), "bg_colour".to_string()];
        let mut b = CommandDoc::new("bg_color", "Set background.");
        b.aliases = vec!["bg_colour".to_string(), "background".to_string()];

        assert_eq!(command_reference(&[a]), command_reference(&[b]));
    }

    #[test]
    fn duplicate_commands_are_collapsed() {
        let docs = vec![
            CommandDoc::new("zoom", "Zoom."),
            CommandDoc::new("zoom", "Zoom again."),
        ];

        let text = command_reference(&docs);

        assert_eq!(text.matches("## zoom").count(), 1);
    }

    #[test]
    fn empty_sections_are_omitted() {
        let text = command_reference(&[CommandDoc::new("pwd", "Print working directory.")]);

        assert!(text.contains("## pwd"));
        assert!(!text.contains("Usage:"));
        assert!(!text.contains("Arguments:"));
        assert!(!text.contains("Example:"));
    }

    #[test]
    fn command_names_include_aliases_and_are_sorted() {
        let names = command_names(&sample());
        assert_eq!(names, vec!["color", "colour", "show"]);
    }

    #[test]
    fn an_empty_registry_still_produces_a_header() {
        let text = command_reference(&[]);
        assert!(text.starts_with("# Patinae command reference"));
    }
}
