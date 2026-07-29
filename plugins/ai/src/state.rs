//! State shared between the panel (which renders it) and the message handler
//! (which drives it).
//!
//! The panel and the handler are separate objects owned by the host, so
//! user intent captured in a panel callback is recorded here as a request flag
//! and acted on by the handler during the next `poll()`.

use std::sync::{Arc, Mutex};

/// One line in the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    User(String),
    Assistant(String),
    /// Tool calls, errors, refusals — anything not authored by either party.
    Note(String),
}

impl Entry {
    pub fn prefix(&self) -> &'static str {
        match self {
            Entry::User(_) => "You: ",
            Entry::Assistant(_) => "Claude: ",
            Entry::Note(_) => "• ",
        }
    }

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
            ..Default::default()
        }
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
            .map(|e| format!("{}{}", e.prefix(), e.body()))
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
    fn mutating_the_transcript_marks_it_dirty() {
        let mut s = SharedState::new();
        s.dirty = false;
        s.push_text_delta("x");
        assert!(s.dirty);
    }
}
