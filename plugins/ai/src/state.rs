//! State shared between the panel (which renders it) and the message handler
//! (which drives it).
//!
//! The panel and the handler are separate objects owned by the host, so
//! user intent captured in a panel callback is recorded here as a request flag
//! and acted on by the handler during the next `poll()`.

use std::sync::{Arc, Mutex};

use patinae_framework::plugin_ui::{PanelMessage, PanelMessageRole, PanelMessageStatus};

use crate::markdown;
use crate::provider::{ModelInfo, ProviderId};

/// How a tool call turned out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Ok,
    Denied,
    Failed,
}

/// One entry in the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    User(String),
    Assistant(String),
    /// Something the agent *did* rather than said.
    ///
    /// Kept structured rather than flattened into a note the moment it happens,
    /// because the outcome arrives later: the command is dispatched, and only a
    /// poll or two afterwards does the host say whether it worked. A row that
    /// can be updated in place is what lets the transcript show a run of
    /// commands and, against each, whether it actually touched the scene.
    Tool {
        /// The provider's tool-call id — how a completed execution finds its
        /// row again.
        call_id: String,
        name: String,
        command: String,
        status: ToolStatus,
    },
    /// Errors, refusals — anything not authored by either party.
    Note(String),
}

impl Entry {
    pub fn body(&self) -> &str {
        match self {
            Entry::User(t) | Entry::Assistant(t) | Entry::Note(t) => t,
            Entry::Tool { command, .. } => command,
        }
    }
}

/// A tool call awaiting the user's approval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pending {
    pub call_id: String,
    pub name: String,
    /// The exact payload that would run, shown verbatim so the user approves
    /// what they can actually see.
    pub payload: String,
}

#[derive(Debug, Default)]
pub struct SharedState {
    pub status: String,
    pub busy: bool,
    pub transcript: Vec<Entry>,
    pub input: String,
    pub pending: Option<Pending>,

    /// How the assistant is labelled in the transcript. Follows the active
    /// provider, because a transcript that says "Claude:" while Gemini is
    /// answering is simply wrong.
    pub assistant_label: String,

    /// The active provider's catalogue, and which provider it describes.
    ///
    /// Tagged so a fetch that lands after the user switched away is discarded
    /// rather than shown against the wrong provider.
    pub models: Vec<ModelInfo>,
    pub models_provider: Option<ProviderId>,

    /// Free text narrowing the model picker. Transient UI state, deliberately
    /// not a setting — nobody wants their filter box restored at startup.
    pub model_filter: String,

    /// Whether the model/provider disclosure under the composer is expanded.
    ///
    /// Collapsed by default: the picker is a decision made about once a
    /// session, and left open it costs more of a narrow dock than the
    /// conversation it sits beneath.
    pub settings_open: bool,

    /// Set by the panel the first time it renders, and never cleared.
    ///
    /// The catalogue is only worth fetching once someone can see the picker.
    /// `poll()` runs from startup whether or not the panel was ever opened, so
    /// without this gate every Patinae launch would build a Tokio runtime and
    /// make a network call for a dropdown nobody is looking at — undoing the
    /// laziness the worker is careful about elsewhere.
    pub panel_shown: bool,

    // --- requests raised by panel callbacks, consumed by the handler ---
    pub submit: Option<String>,
    pub decision: Option<bool>,
    pub cancel_requested: bool,
    pub reset_requested: bool,
    pub login_requested: bool,
    pub logout_requested: bool,

    /// Set whenever the rendered content changed, so the handler knows to ask
    /// the host for a repaint.
    pub dirty: bool,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            status: "Checking sign-in…".to_string(),
            assistant_label: ProviderId::Claude.display_name().to_string(),
            ..Default::default()
        }
    }

    /// The name an entry is rendered under.
    ///
    /// Lives here rather than on [`Entry`] because the assistant's name depends
    /// on which provider is active, which an entry does not know.
    pub fn author(&self, entry: &Entry) -> String {
        match entry {
            Entry::User(_) => "You".to_string(),
            Entry::Assistant(_) => self.assistant_label.clone(),
            Entry::Tool { name, .. } => name.clone(),
            Entry::Note(_) => "Note".to_string(),
        }
    }

    /// Mark the tool row belonging to `call_id` with its outcome.
    ///
    /// A miss is not an error: a command can be dispatched by a path that never
    /// wrote a row, and a transcript cleared mid-flight legitimately loses the
    /// row the result was coming back to.
    pub fn resolve_tool(&mut self, call_id: &str, outcome: ToolStatus) {
        for entry in self.transcript.iter_mut().rev() {
            if let Entry::Tool {
                call_id: id,
                status,
                ..
            } = entry
            {
                if id == call_id {
                    *status = outcome;
                    self.dirty = true;
                    return;
                }
            }
        }
    }

    /// Capability metadata for `id`, if the catalogue describes it.
    pub fn model_info(&self, id: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Append streamed assistant text, merging into the current assistant entry
    /// so a turn reads as one paragraph rather than one line per token.
    pub fn push_text_delta(&mut self, text: &str) {
        match self.transcript.last_mut() {
            Some(Entry::Assistant(existing)) => existing.push_str(text),
            _ => self.transcript.push(Entry::Assistant(text.to_string())),
        }
        self.dirty = true;
    }

    pub fn push(&mut self, entry: Entry) {
        self.transcript.push(entry);
        self.dirty = true;
    }

    /// Build the transcript's messages for the panel.
    ///
    /// Ids are positional. Entries are only ever appended to — a streamed delta
    /// merges into the last one rather than adding a row — so an index is
    /// stable for the life of a conversation, and clearing resets the whole
    /// model anyway.
    pub fn messages(&self) -> Vec<PanelMessage> {
        self.transcript
            .iter()
            .enumerate()
            .map(|(i, entry)| self.message(i, entry))
            .collect()
    }

    fn message(&self, index: usize, entry: &Entry) -> PanelMessage {
        let id = format!("m{index}");
        let author = self.author(entry);
        match entry {
            Entry::User(text) => {
                PanelMessage::text(id, PanelMessageRole::User, author, text.clone())
            }
            // Only the assistant writes markdown, so only the assistant pays
            // for splitting it.
            Entry::Assistant(text) => PanelMessage::new(
                id,
                PanelMessageRole::Assistant,
                author,
                markdown::blocks(text),
            ),
            Entry::Tool {
                command, status, ..
            } => PanelMessage::new(id, PanelMessageRole::Tool, author, Vec::new())
                .status(match status {
                    ToolStatus::Running => PanelMessageStatus::Running,
                    ToolStatus::Ok => PanelMessageStatus::Ok,
                    ToolStatus::Denied => PanelMessageStatus::Denied,
                    ToolStatus::Failed => PanelMessageStatus::Failed,
                })
                .detail(command.clone()),
            Entry::Note(text) => {
                PanelMessage::text(id, PanelMessageRole::Error, author, text.clone())
            }
        }
    }
}

/// What the transcript shows before the first exchange.
///
/// An agent whose capabilities are invisible is an agent nobody tries, so the
/// empty state spends its space on two prompts worth copying rather than on
/// describing itself.
pub const ONBOARDING: &str = "Ask the agent to do something with the structure — for example:\n\
     \n    load 1crn and show it as cartoon coloured by chain\n\
     \n    what chains are in this structure and how long are they?";

/// Convenience alias — both the panel and the handler hold one of these.
pub type Shared = Arc<Mutex<SharedState>>;

pub fn new_shared() -> Shared {
    Arc::new(Mutex::new(SharedState::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_deltas_merge_into_one_assistant_entry() {
        let mut s = SharedState::new();
        s.push_text_delta("Hello ");
        s.push_text_delta("world");
        assert_eq!(s.transcript, vec![Entry::Assistant("Hello world".into())]);
    }

    #[test]
    fn a_tool_call_breaks_the_merge_run() {
        // Text after a tool call is a new paragraph, not a continuation of the
        // sentence the agent was in the middle of.
        let mut s = SharedState::new();
        s.push_text_delta("first");
        s.push(tool_call("t1", "load 1crn"));
        s.push_text_delta("second");
        assert_eq!(s.transcript.len(), 3);
        assert_eq!(s.transcript[2], Entry::Assistant("second".into()));
    }

    fn tool_call(call_id: &str, command: &str) -> Entry {
        Entry::Tool {
            call_id: call_id.into(),
            name: "run_command".into(),
            command: command.into(),
            status: ToolStatus::Running,
        }
    }

    #[test]
    fn the_onboarding_copy_offers_something_worth_copying() {
        assert!(ONBOARDING.contains("load 1crn"));
    }

    #[test]
    fn each_entry_becomes_a_message_with_its_own_role() {
        let mut s = SharedState::new();
        s.push(Entry::User("hi".into()));
        s.push(Entry::Assistant("hello".into()));
        s.push(tool_call("t1", "load 1crn"));

        let messages = s.messages();
        assert_eq!(messages[0].role, PanelMessageRole::User);
        assert_eq!(messages[0].author, "You");
        assert_eq!(messages[1].role, PanelMessageRole::Assistant);
        assert_eq!(messages[1].author, "Claude");
        assert_eq!(messages[2].role, PanelMessageRole::Tool);
        assert_eq!(messages[2].author, "run_command");
        assert_eq!(messages[2].detail, "load 1crn");
    }

    #[test]
    fn message_ids_are_positional_and_stable_across_a_streamed_delta() {
        // The renderer reuses rows by id, so a delta must not renumber them.
        let mut s = SharedState::new();
        s.push(Entry::User("hi".into()));
        s.push_text_delta("hel");
        let before: Vec<String> = s.messages().into_iter().map(|m| m.id).collect();

        s.push_text_delta("lo");
        let after: Vec<String> = s.messages().into_iter().map(|m| m.id).collect();
        assert_eq!(before, after);
        assert_eq!(after, ["m0", "m1"]);
    }

    #[test]
    fn the_assistant_is_labelled_with_whichever_provider_is_answering() {
        // A transcript that says "Claude" while Gemini answers is just wrong.
        let mut s = SharedState::new();
        s.assistant_label = "Gemini".into();
        s.push(Entry::Assistant("hello".into()));
        assert_eq!(s.messages()[0].author, "Gemini");
    }

    #[test]
    fn an_assistant_reply_is_split_into_prose_and_code() {
        let mut s = SharedState::new();
        s.push(Entry::Assistant("Run:\n```\nload 1crn\n```".into()));
        let blocks = &s.messages()[0].blocks;
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1].text(), "load 1crn");
    }

    #[test]
    fn a_users_own_words_are_never_parsed_as_markdown() {
        // Backticks the user typed are just characters they typed.
        let mut s = SharedState::new();
        s.push(Entry::User("what does ```load``` do?".into()));
        let blocks = &s.messages()[0].blocks;
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text(), "what does ```load``` do?");
    }

    #[test]
    fn a_tool_row_is_resolved_in_place_by_its_call_id() {
        let mut s = SharedState::new();
        s.push(tool_call("t1", "load 1crn"));
        s.push(tool_call("t2", "show cartoon"));

        s.resolve_tool("t2", ToolStatus::Failed);
        s.resolve_tool("t1", ToolStatus::Ok);

        assert_eq!(s.messages()[0].status, PanelMessageStatus::Ok);
        assert_eq!(s.messages()[1].status, PanelMessageStatus::Failed);
        // Resolving must not add rows — the outcome updates the call.
        assert_eq!(s.transcript.len(), 2);
    }

    #[test]
    fn resolving_a_tool_row_that_is_gone_is_harmless() {
        // Clearing the transcript mid-flight legitimately loses the row a
        // pending result was coming back to.
        let mut s = SharedState::new();
        s.resolve_tool("t1", ToolStatus::Ok);
        assert!(s.transcript.is_empty());
    }

    #[test]
    fn model_info_is_looked_up_by_id() {
        let mut s = SharedState::new();
        s.models = vec![crate::provider::ModelInfo::new("a/b", "A B")];
        assert_eq!(s.model_info("a/b").map(|m| m.label.as_str()), Some("A B"));
        assert!(s.model_info("nope").is_none());
    }

    #[test]
    fn mutating_the_transcript_marks_it_dirty() {
        let mut s = SharedState::new();
        s.dirty = false;
        s.push_text_delta("x");
        assert!(s.dirty);
    }
}
