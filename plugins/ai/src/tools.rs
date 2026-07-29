//! Tool definitions and their main-thread implementations.
//!
//! Tool *schemas* are pure data and can be built anywhere. Tool *execution*
//! splits in two:
//!
//! - **Deferred** (`run_command`, `run_python`) go through
//!   `PollContext::execute_command`; the result arrives in a later `poll()`.
//! - **Immediate** (everything else) run inside a `queue_viewer_mutation`
//!   closure on the main thread, where a live `&mut dyn ViewerLike` is
//!   available. That call returns nothing, so results are sent back over a
//!   channel captured by the closure.
//!
//! Descriptions state *when* to call a tool, not just what it does — recent
//! models reach for tools conservatively, and prescriptive descriptions
//! measurably raise the call rate.

use patinae_framework::model::sequence::{
    chain_to_fasta, collect_all_sequences, SequenceColorContext, SequenceModel,
};
use patinae_plugin::prelude::{select, ViewerLike};
use serde_json::{json, Value};

use crate::provider::ToolSpec;

pub const RUN_COMMAND: &str = "run_command";
pub const RUN_PYTHON: &str = "run_python";
pub const SCREENSHOT: &str = "screenshot";
pub const GET_SCENE_STATE: &str = "get_scene_state";
pub const GET_SEQUENCES: &str = "get_sequences";
pub const COUNT_ATOMS: &str = "count_atoms";

/// Tools that mutate the scene and therefore pass through approval gating.
pub fn is_mutating(name: &str) -> bool {
    matches!(name, RUN_COMMAND | RUN_PYTHON)
}

/// Build the tool schema list.
///
/// `allow_python` drops `run_python` entirely when the setting is off or the
/// `python-plugin` is not installed — a tool that always errors is worse than
/// one Claude never sees.
pub fn schemas(allow_python: bool) -> Vec<ToolSpec> {
    let mut tools = vec![
        ToolSpec {
            name: RUN_COMMAND.to_string(),
            description: "Run one or more Patinae viewer commands. This is your primary way to \
                 act on the scene: loading structures, changing representations, colouring, \
                 selecting, moving the camera, saving images. Separate multiple commands with \
                 ';'. Call this whenever the user asks you to change what is displayed."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "commands": {
                        "type": "string",
                        "description": "Patinae command(s), e.g. \"load 1crn; show cartoon; color green, chain A\""
                    }
                },
                "required": ["commands"],
            }),
        },
        ToolSpec {
            name: SCREENSHOT.to_string(),
            description: "Capture the current viewport and return it as an image. Call this to \
                 check your own work after changing the scene, to answer questions about how \
                 something looks, or when the user asks what is on screen. Note that selection \
                 and hover highlights are NOT included in captures — to see a selection, colour \
                 it or use 'indicate' first."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "width":  {"type": "integer", "description": "Optional width in pixels"},
                    "height": {"type": "integer", "description": "Optional height in pixels"}
                },
            }),
        },
        ToolSpec {
            name: GET_SCENE_STATE.to_string(),
            description: "List what is currently loaded: object names and types, which are \
                 enabled, atom counts, named selections and their expressions, and the viewport \
                 size. Call this before acting when you are unsure what is in the scene, rather \
                 than guessing object names."
                .to_string(),
            input_schema: json!({"type": "object", "properties": {}}),
        },
        ToolSpec {
            name: GET_SEQUENCES.to_string(),
            description: "Get residue sequences in FASTA format for loaded structures. Call this \
                 for any question about sequence, chain composition, chain lengths, or residue \
                 numbering, and before constructing a residue-range selection."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "object": {"type": "string", "description": "Limit to one object (optional)"},
                    "chain":  {"type": "string", "description": "Limit to one chain (optional)"}
                },
            }),
        },
        ToolSpec {
            name: COUNT_ATOMS.to_string(),
            description: "Count the atoms matching a selection expression. This is cheap — call \
                 it to validate a selection expression before using it in a command, especially \
                 for complex expressions, so you catch a typo or an empty match early."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "selection": {
                        "type": "string",
                        "description": "Selection expression, e.g. \"chain A and name CA\""
                    }
                },
                "required": ["selection"],
            }),
        },
    ];

    if allow_python {
        tools.push(ToolSpec {
            name: RUN_PYTHON.to_string(),
            description: "Execute Python inside the viewer's embedded interpreter, where `cmd` \
                 is the Patinae command API. Use this when a task needs logic that plain \
                 commands cannot express — loops, conditionals, arithmetic over atoms, or \
                 analysis across many residues. For a simple sequence of viewer actions, prefer \
                 run_command."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute"}
                },
                "required": ["code"],
            }),
        });
    }

    tools
}

/// Translate a tool call into the command string to hand to the executor.
///
/// Returns `None` for tools that are not command-backed.
pub fn command_for(name: &str, input: &Value) -> Option<String> {
    match name {
        RUN_COMMAND => input
            .get("commands")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        // Routed through the python-plugin's own `python` command rather than
        // linking against it, so this plugin has no build-time dependency on it.
        RUN_PYTHON => input
            .get("code")
            .and_then(|v| v.as_str())
            .map(|code| format!("python {code}")),
        _ => None,
    }
}

// =============================================================================
// Immediate (viewer-reading) tools
// =============================================================================

/// Human-readable summary of everything currently loaded.
pub fn scene_state(viewer: &dyn ViewerLike) -> String {
    let registry = viewer.objects();
    let enabled: Vec<String> = registry
        .enabled_objects()
        .map(|o| o.name().to_string())
        .collect();

    let mut out = String::new();
    out.push_str("# Objects\n");
    let names: Vec<String> = registry.names().map(|n| n.to_string()).collect();
    if names.is_empty() {
        out.push_str("(none loaded)\n");
    } else {
        for name in &names {
            let kind = registry
                .get(name)
                .map(|o| format!("{:?}", o.object_type()))
                .unwrap_or_else(|| "Unknown".to_string());
            let atoms = registry
                .get_molecule(name)
                .map(|m| format!(", {} atoms", m.molecule().atom_count()))
                .unwrap_or_default();
            let state = if enabled.contains(name) {
                "enabled"
            } else {
                "disabled"
            };
            out.push_str(&format!("- {name} ({kind}, {state}{atoms})\n"));
        }
    }

    out.push_str("\n# Selections\n");
    let selections = viewer.selections().names();
    if selections.is_empty() {
        out.push_str("(none defined)\n");
    } else {
        for name in selections {
            let expr = viewer
                .selections()
                .get_expression(&name)
                .unwrap_or("<unknown>");
            out.push_str(&format!("- {name}: {expr}\n"));
        }
    }

    let (w, h) = viewer.viewport_size();
    out.push_str(&format!("\n# Viewport\n{w}x{h}\n"));

    // Objects carry no source-path provenance in the scene graph, so be honest
    // rather than inventing file names from object names.
    out.push_str(
        "\nNote: object names usually derive from the source filename, but the scene \
         does not record file paths, so these are not guaranteed to be paths.\n",
    );
    out
}

/// FASTA for loaded polymers, optionally narrowed to one object and/or chain.
pub fn sequences(viewer: &dyn ViewerLike, object: Option<&str>, chain: Option<&str>) -> String {
    let session = viewer.session();
    let ctx = SequenceColorContext {
        named_palette: &session.named_palette,
        palette: &session.palette,
    };
    let mut model = SequenceModel::new();
    model.rebuild_cache(viewer.objects(), &ctx);

    if object.is_none() && chain.is_none() {
        let (fasta, count) = collect_all_sequences(&model.sequences);
        return if count == 0 {
            "(no polymer sequences — nothing loaded, or only non-polymer objects)".to_string()
        } else {
            fasta
        };
    }

    let mut out = String::new();
    for seq_object in &model.sequences {
        if let Some(want) = object {
            if seq_object.object_name != want {
                continue;
            }
        }
        for seq_chain in &seq_object.chains {
            if let Some(want) = chain {
                if seq_chain.chain_id != want {
                    continue;
                }
            }
            if let Some((fasta, _)) = chain_to_fasta(&seq_object.object_name, seq_chain) {
                out.push_str(&fasta);
                out.push('\n');
            }
        }
    }

    if out.is_empty() {
        "(no sequences matched that object/chain filter)".to_string()
    } else {
        out
    }
}

/// Count atoms matching a selection expression across every loaded molecule.
pub fn count_atoms(viewer: &dyn ViewerLike, expression: &str) -> String {
    let registry = viewer.objects();
    let names: Vec<String> = registry.names().map(|n| n.to_string()).collect();
    if names.is_empty() {
        return "0 atoms (nothing is loaded)".to_string();
    }

    let mut total = 0usize;
    let mut per_object: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    for name in &names {
        let Some(mol_obj) = registry.get_molecule(name) else {
            continue;
        };
        match select(mol_obj.molecule(), expression) {
            Ok(result) => {
                let n = result.count();
                total += n;
                if n > 0 {
                    per_object.push(format!("  {name}: {n}"));
                }
            }
            Err(e) => errors.push(format!("  {name}: {e}")),
        }
    }

    if !errors.is_empty() && total == 0 {
        return format!(
            "Selection `{expression}` failed to evaluate:\n{}",
            errors.join("\n")
        );
    }

    let mut out = format!("{total} atoms match `{expression}`");
    if !per_object.is_empty() {
        out.push('\n');
        out.push_str(&per_object.join("\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_tool_is_hidden_when_disallowed() {
        let names: Vec<String> = schemas(false).into_iter().map(|t| t.name).collect();
        assert!(!names.contains(&RUN_PYTHON.to_string()));
        assert!(names.contains(&RUN_COMMAND.to_string()));
    }

    #[test]
    fn python_tool_appears_when_allowed() {
        let names: Vec<String> = schemas(true).into_iter().map(|t| t.name).collect();
        assert!(names.contains(&RUN_PYTHON.to_string()));
    }

    #[test]
    fn every_schema_is_a_valid_object_schema() {
        for tool in schemas(true) {
            assert_eq!(tool.input_schema["type"], "object", "{}", tool.name);
            assert!(
                tool.input_schema.get("properties").is_some(),
                "{} has no properties",
                tool.name
            );
            assert!(!tool.description.is_empty());
        }
    }

    #[test]
    fn only_scene_mutating_tools_are_gated() {
        assert!(is_mutating(RUN_COMMAND));
        assert!(is_mutating(RUN_PYTHON));
        for read_only in [SCREENSHOT, GET_SCENE_STATE, GET_SEQUENCES, COUNT_ATOMS] {
            assert!(!is_mutating(read_only), "{read_only} should not be gated");
        }
    }

    #[test]
    fn run_command_maps_to_the_raw_command_string() {
        let input = json!({"commands": "load 1crn; show cartoon"});
        assert_eq!(
            command_for(RUN_COMMAND, &input).as_deref(),
            Some("load 1crn; show cartoon")
        );
    }

    #[test]
    fn run_python_is_routed_through_the_python_command() {
        let input = json!({"code": "print(cmd.get_names())"});
        assert_eq!(
            command_for(RUN_PYTHON, &input).as_deref(),
            Some("python print(cmd.get_names())")
        );
    }

    #[test]
    fn non_command_tools_have_no_command_form() {
        assert!(command_for(SCREENSHOT, &json!({})).is_none());
        assert!(command_for(GET_SCENE_STATE, &json!({})).is_none());
    }

    #[test]
    fn missing_required_argument_yields_no_command() {
        assert!(command_for(RUN_COMMAND, &json!({})).is_none());
        assert!(command_for(RUN_PYTHON, &json!({"wrong": "key"})).is_none());
    }
}
