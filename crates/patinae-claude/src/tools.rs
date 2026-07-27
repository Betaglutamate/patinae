//! Tool definitions exposed to the model.
//!
//! Tools are *data* here. This crate declares their names and schemas; the
//! host executes them against the live `AppKernel`. Keeping execution out of
//! the crate is what lets the agent core stay free of scene and render
//! dependencies.

use serde::{Deserialize, Serialize};

use crate::types::{ServerTool, Tool, ToolDefinition};

/// Runs a Patinae command and returns its output.
pub const RUN_COMMAND: &str = "run_command";
/// Lists loaded objects and selections.
pub const LIST_OBJECTS: &str = "list_objects";
/// Counts atoms matching a selection expression.
pub const COUNT_ATOMS: &str = "count_atoms";
/// Reads the current camera view.
pub const GET_CAMERA_VIEW: &str = "get_camera_view";
/// Reads a setting value.
pub const GET_SETTING: &str = "get_setting";
/// Captures the rendered viewport as an image.
pub const SCREENSHOT_VIEWPORT: &str = "screenshot_viewport";
/// Looks up RCSB metadata for a PDB entry.
pub const FETCH_STRUCTURE_METADATA: &str = "fetch_structure_metadata";

/// Versioned type for the server-side web search tool.
const WEB_SEARCH_TYPE: &str = "web_search_20260209";

/// A tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Correlation id to echo back on the result.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Parsed input.
    pub input: serde_json::Value,
}

impl ToolCall {
    /// Returns a string field from the tool input.
    pub fn string_arg(&self, key: &str) -> Option<&str> {
        self.input.get(key)?.as_str()
    }

    /// Returns the command a [`RUN_COMMAND`] call wants to execute.
    pub fn command(&self) -> Option<&str> {
        self.string_arg("command")
    }
}

/// The outcome of executing a tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutcome {
    /// Correlation id of the originating call.
    pub tool_use_id: String,
    /// Result text handed back to the model.
    pub content: String,
    /// Whether the tool failed. Failures are reported, never dropped, so the
    /// model can adapt instead of stalling.
    pub is_error: bool,
    /// Optional base64 PNG returned alongside the text.
    pub image_png_base64: Option<String>,
}

impl ToolOutcome {
    /// Builds a successful outcome.
    pub fn ok(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: false,
            image_png_base64: None,
        }
    }

    /// Builds a failed outcome.
    pub fn error(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: true,
            image_png_base64: None,
        }
    }

    /// Attaches a rendered image to the outcome.
    pub fn with_image(mut self, png_base64: impl Into<String>) -> Self {
        self.image_png_base64 = Some(png_base64.into());
        self
    }
}

/// Which optional capabilities are available to the model this session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolCapabilities {
    /// Whether the viewport can be captured (needs a live renderer).
    pub screenshots: bool,
    /// Whether RCSB lookups are permitted.
    pub structure_lookup: bool,
    /// Whether the server-side web search tool is enabled.
    pub web_search: bool,
}

impl Default for ToolCapabilities {
    fn default() -> Self {
        Self {
            screenshots: true,
            structure_lookup: true,
            web_search: true,
        }
    }
}

/// Builds the tool list for a request.
///
/// Order is fixed rather than derived, because `tools` renders at the very
/// front of the prompt: reordering between turns invalidates the entire cache.
pub fn tool_definitions(capabilities: ToolCapabilities) -> Vec<Tool> {
    let mut tools = vec![
        Tool::Custom(run_command_tool()),
        Tool::Custom(no_argument_tool(
            LIST_OBJECTS,
            "List loaded objects, their representations, and named selections. \
             Call this before acting when you are unsure what is in the scene.",
        )),
        Tool::Custom(single_string_tool(
            COUNT_ATOMS,
            "Count atoms matching a Patinae selection expression. Call this to \
             check a selection is non-empty before building on it.",
            "selection",
            "A selection expression, e.g. `chain A and name CA`.",
        )),
        Tool::Custom(no_argument_tool(
            GET_CAMERA_VIEW,
            "Read the current camera view as the 18-value PyMOL-style matrix.",
        )),
        Tool::Custom(single_string_tool(
            GET_SETTING,
            "Read the current value of a Patinae setting.",
            "name",
            "The setting name, e.g. `sphere_scale`.",
        )),
    ];

    if capabilities.screenshots {
        tools.push(Tool::Custom(no_argument_tool(
            SCREENSHOT_VIEWPORT,
            "Capture what is currently rendered in the viewport and return it \
             as an image. Call this when the user asks about appearance, or to \
             check that a visual change landed as intended.",
        )));
    }
    if capabilities.structure_lookup {
        tools.push(Tool::Custom(single_string_tool(
            FETCH_STRUCTURE_METADATA,
            "Look up title, experimental method, and release dates for a PDB \
             entry from RCSB. This does not load the structure.",
            "pdb_id",
            "A four-character PDB identifier, e.g. `1ubq`.",
        )));
    }
    if capabilities.web_search {
        tools.push(Tool::Server(ServerTool {
            kind: WEB_SEARCH_TYPE,
            name: "web_search",
        }));
    }
    tools
}

/// The command-execution tool.
///
/// `strict` is set so the input always validates: a malformed `command` field
/// would otherwise reach the risk classifier as `None` and be treated as an
/// unknown command.
fn run_command_tool() -> ToolDefinition {
    ToolDefinition {
        name: RUN_COMMAND.to_string(),
        description: "Run a Patinae viewer command and return its output. This is the \
             primary way to change the scene. Commands use PyMOL-style syntax, \
             e.g. `show cartoon, chain A`. Prefer one command per call so a \
             failure is easy to attribute."
            .to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to run, e.g. `show cartoon, chain A`."
                }
            },
            "required": ["command"],
            "additionalProperties": false
        }),
        strict: true,
    }
}

/// Builds a tool taking no arguments.
fn no_argument_tool(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        }),
        strict: true,
    }
}

/// Builds a tool taking a single required string argument.
fn single_string_tool(
    name: &str,
    description: &str,
    arg: &str,
    arg_description: &str,
) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                arg: { "type": "string", "description": arg_description }
            },
            "required": [arg],
            "additionalProperties": false
        }),
        strict: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(tools: &[Tool]) -> Vec<String> {
        tools
            .iter()
            .map(|tool| match tool {
                Tool::Custom(def) => def.name.clone(),
                Tool::Server(server) => server.name.to_string(),
            })
            .collect()
    }

    #[test]
    fn all_capabilities_produce_the_full_tool_set() {
        let tools = tool_definitions(ToolCapabilities::default());

        assert_eq!(
            names(&tools),
            vec![
                RUN_COMMAND,
                LIST_OBJECTS,
                COUNT_ATOMS,
                GET_CAMERA_VIEW,
                GET_SETTING,
                SCREENSHOT_VIEWPORT,
                FETCH_STRUCTURE_METADATA,
                "web_search",
            ]
        );
    }

    #[test]
    fn disabled_capabilities_drop_their_tools() {
        let tools = tool_definitions(ToolCapabilities {
            screenshots: false,
            structure_lookup: false,
            web_search: false,
        });

        let names = names(&tools);
        assert!(!names.contains(&SCREENSHOT_VIEWPORT.to_string()));
        assert!(!names.contains(&FETCH_STRUCTURE_METADATA.to_string()));
        assert!(!names.contains(&"web_search".to_string()));
        // The core tools are always present.
        assert!(names.contains(&RUN_COMMAND.to_string()));
    }

    #[test]
    fn tool_order_is_stable_across_calls() {
        // `tools` renders at the front of the prompt; reordering between turns
        // would invalidate the whole cache.
        let first = tool_definitions(ToolCapabilities::default());
        let second = tool_definitions(ToolCapabilities::default());

        assert_eq!(
            serde_json::to_string(&first).expect("tools should serialize"),
            serde_json::to_string(&second).expect("tools should serialize")
        );
    }

    #[test]
    fn custom_tools_serialize_with_a_schema_and_server_tools_with_a_type() {
        let tools = tool_definitions(ToolCapabilities::default());
        let json = serde_json::to_value(&tools).expect("tools should serialize");
        let array = json.as_array().expect("tools should be an array");

        let run = &array[0];
        assert_eq!(run["name"], RUN_COMMAND);
        assert_eq!(run["input_schema"]["type"], "object");
        assert_eq!(run["strict"], true);
        assert!(run.get("type").is_none(), "custom tools carry no type");

        let search = array.last().expect("web search should be last");
        assert_eq!(search["type"], WEB_SEARCH_TYPE);
        assert_eq!(search["name"], "web_search");
        assert!(
            search.get("input_schema").is_none(),
            "server tools carry no schema"
        );
    }

    #[test]
    fn run_command_schema_is_strict_and_closed() {
        let tool = run_command_tool();

        assert!(tool.strict);
        assert_eq!(tool.input_schema["additionalProperties"], false);
        assert_eq!(tool.input_schema["required"][0], "command");
    }

    #[test]
    fn tool_call_extracts_the_command_argument() {
        let call = ToolCall {
            id: "toolu_1".to_string(),
            name: RUN_COMMAND.to_string(),
            input: serde_json::json!({"command": "show cartoon"}),
        };

        assert_eq!(call.command(), Some("show cartoon"));
        assert_eq!(call.string_arg("missing"), None);
    }

    #[test]
    fn a_non_string_command_argument_reads_as_absent() {
        // Must not panic or coerce; the caller treats None as unusable input.
        let call = ToolCall {
            id: "toolu_1".to_string(),
            name: RUN_COMMAND.to_string(),
            input: serde_json::json!({"command": 42}),
        };

        assert_eq!(call.command(), None);
    }

    #[test]
    fn outcomes_carry_success_failure_and_images() {
        assert!(!ToolOutcome::ok("t", "fine").is_error);
        assert!(ToolOutcome::error("t", "bad").is_error);

        let with_image = ToolOutcome::ok("t", "here").with_image("QUJD");
        assert_eq!(with_image.image_png_base64.as_deref(), Some("QUJD"));
    }
}
