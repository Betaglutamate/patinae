//! State shared between the panel (which renders it) and the message handler
//! (which drives it).
//!
//! The panel and the handler are separate objects owned by the host, so
//! user intent captured in a panel callback is recorded here as a request flag
//! and acted on by the handler during the next `poll()`.

use std::sync::{Arc, Mutex};

use crate::provider::{ModelInfo, ProviderId};

/// One line in the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    User(String),
    Assistant(String),
    /// Tool calls, errors, refusals — anything not authored by either party.
    Note(String),
}

impl Entry {
    pub fn body(&self) -> &str {
        match self {
            Entry::User(t) | Entry::Assistant(t) | Entry::Note(t) => t,
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

    /// The label an entry is rendered behind.
    ///
    /// Lives here rather than on [`Entry`] because the assistant's name depends
    /// on which provider is active, which an entry does not know.
    pub fn prefix(&self, entry: &Entry) -> String {
        match entry {
            Entry::User(_) => "You: ".to_string(),
            Entry::Assistant(_) => format!("{}: ", self.assistant_label),
            Entry::Note(_) => "\u{2022} ".to_string(),
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

    /// Render the transcript for the panel's text area.
    pub fn render(&self) -> String {
        if self.transcript.is_empty() {
            return "Ask the agent to do something with the structure — for example:\n\
                    \n  load 1crn and show it as cartoon coloured by chain\n\
                    \n  what chains are in this structure and how long are they?"
                .to_string();
        }
        self.transcript
            .iter()
            .map(|e| format!("{}{}", self.prefix(e), e.body()))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

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
    fn a_note_breaks_the_merge_run() {
        let mut s = SharedState::new();
        s.push_text_delta("first");
        s.push(Entry::Note("ran run_command".into()));
        s.push_text_delta("second");
        assert_eq!(s.transcript.len(), 3);
        assert_eq!(s.transcript[2], Entry::Assistant("second".into()));
    }

    #[test]
    fn empty_transcript_renders_onboarding_help() {
        let s = SharedState::new();
        assert!(s.render().contains("load 1crn"));
    }

    #[test]
    fn rendered_transcript_labels_each_speaker() {
        let mut s = SharedState::new();
        s.push(Entry::User("hi".into()));
        s.push(Entry::Assistant("hello".into()));
        let out = s.render();
        assert!(out.contains("You: hi"));
        assert!(out.contains("Claude: hello"));
    }

    #[test]
    fn the_assistant_is_labelled_with_whichever_provider_is_answering() {
        // A transcript that says "Claude:" while Gemini answers is just wrong.
        let mut s = SharedState::new();
        s.assistant_label = "Gemini".into();
        s.push(Entry::Assistant("hello".into()));
        assert!(s.render().contains("Gemini: hello"));
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
