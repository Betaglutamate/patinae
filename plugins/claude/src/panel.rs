//! The chat panel, docked at the bottom alongside the Python script editor.
//!
//! Plugin panels are declarative: this returns a `PanelSnapshot` and the
//! frontend renders it, so nothing here touches Slint. Two constraints shape
//! the layout — the Slint bridge flattens nesting to four levels, and panel
//! callbacks cannot reach the viewer. User intent is therefore recorded as a
//! request flag in [`SharedState`] and acted on by the handler during `poll()`.

use patinae_plugin::prelude::*;

use crate::state::{Entry, Shared, SharedState};

const PANEL_ID: &str = "claude_chat";

// Control ids, also used to route events back.
const STATUS: &str = "status";
const AUTH_ACTION: &str = "auth_action";
const TRANSCRIPT: &str = "transcript";
const PROMPT_INPUT: &str = "prompt";
const ACTIONS: &str = "actions";
const SEND: &str = "send";
const STOP: &str = "stop";
const CLEAR: &str = "clear";
const APPROVAL: &str = "approval";
const APPROVE_ROW: &str = "approve_row";
const ALLOW: &str = "allow";
const DENY: &str = "deny";

pub struct ChatPanel {
    state: Shared,
}

impl ChatPanel {
    pub fn new(state: Shared) -> Self {
        Self { state }
    }
}

/// Highlight each speaker label so a long transcript stays scannable.
///
/// Ranges are byte offsets into the rendered text, matching the order
/// [`SharedState::render`] emits entries.
fn speaker_highlights(state: &SharedState) -> Vec<PanelTextHighlight> {
    let mut highlights = Vec::new();
    let mut offset = 0usize;
    for (i, entry) in state.transcript.iter().enumerate() {
        if i > 0 {
            offset += 2; // the "\n\n" separator
        }
        let prefix = entry.prefix();
        let style = match entry {
            Entry::User(_) => PanelTextStyle::Keyword,
            Entry::Assistant(_) => PanelTextStyle::Function,
            Entry::Note(_) => PanelTextStyle::Comment,
        };
        highlights.push(PanelTextHighlight::new(
            offset,
            offset + prefix.len(),
            style,
        ));
        offset += prefix.len() + entry.body().len();
    }
    highlights
}

impl PluginPanel for ChatPanel {
    fn descriptor(&self) -> PanelDescriptor {
        PanelDescriptor::bottom(PANEL_ID, "Claude")
            .icon("AI")
            .default_visible(false)
    }

    fn runtime_requirements(&self) -> PanelRuntimeRequirements {
        // The panel renders plugin-owned state only, so there is no reason to
        // pay for a serialized session snapshot every frame.
        PanelRuntimeRequirements::NONE
    }

    fn snapshot(&mut self, _ctx: &SharedContext<'_>) -> PanelSnapshot {
        match self.state.lock() {
            Ok(state) => build_snapshot(&state),
            Err(_) => PanelSnapshot::new(vec![PanelControl::Text {
                id: STATUS.into(),
                text: "Claude panel state is unavailable.".into(),
            }]),
        }
    }

    fn handle_event(
        &mut self,
        event: PanelEvent,
        _ctx: &SharedContext<'_>,
        _bus: &mut MessageBus,
    ) -> Vec<PanelAction> {
        if let Ok(mut state) = self.state.lock() {
            apply_event(&mut state, &event);
        }
        // All work is deferred to the handler's poll, which is the only place
        // with viewer access — nothing to hand back to the host here.
        Vec::new()
    }
}

/// Build the panel contents from state. Pure, so it is directly testable.
fn build_snapshot(state: &SharedState) -> PanelSnapshot {
    {
        let signed_in = state.status.starts_with("Signed in");
        let mut controls = vec![
            PanelControl::Text {
                id: STATUS.into(),
                text: state.status.clone(),
            },
            PanelControl::Button {
                id: AUTH_ACTION.into(),
                label: if signed_in { "Sign out" } else { "Sign in" }.into(),
                primary: !signed_in,
            },
            PanelControl::TextArea(
                PanelTextArea::new(
                    TRANSCRIPT,
                    "",
                    state.render(),
                    "",
                    14,
                    true, // read-only: this is a transcript, not an editor
                )
                .with_highlights(speaker_highlights(state)),
            ),
        ];

        // Approval gate. Shown only when a tool call is waiting, so the panel
        // stays quiet in auto-approve mode.
        if let Some(pending) = &state.pending {
            controls.push(PanelControl::Group(PanelGroup::new(
                APPROVAL,
                format!("Allow {}?", pending.name),
                vec![
                    PanelControlNode::new(PanelControl::Text {
                        id: "approval_payload".into(),
                        text: pending.payload.clone(),
                    }),
                    PanelControlNode::new(PanelControl::ButtonRow {
                        id: APPROVE_ROW.into(),
                        buttons: vec![
                            PanelButton::new(ALLOW, "Allow", "", true),
                            PanelButton::new(DENY, "Deny", "", false),
                        ],
                    }),
                ],
            )));
        }

        controls.push(PanelControl::TextInput {
            id: PROMPT_INPUT.into(),
            label: "".into(),
            value: state.input.clone(),
            placeholder: "Ask Claude to do something…".into(),
        });
        controls.push(PanelControl::ButtonRow {
            id: ACTIONS.into(),
            buttons: vec![
                PanelButton::new(SEND, "Send", "", true)
                    .enabled(signed_in && !state.busy && !state.input.trim().is_empty()),
                PanelButton::new(STOP, "Stop", "", false).enabled(state.busy),
                PanelButton::new(CLEAR, "Clear", "", false).enabled(!state.transcript.is_empty()),
            ],
        });

        PanelSnapshot::new(controls)
    }
}

/// Apply one panel event to state. Pure, so it is directly testable.
fn apply_event(state: &mut SharedState, event: &PanelEvent) {
    {
        match event.control_id.as_str() {
            PROMPT_INPUT => {
                if let PanelValue::Text(text) = &event.value {
                    let text = text.clone();
                    // A commit (Enter) both sets and submits; a plain edit only
                    // records the draft.
                    if matches!(event.kind, PanelEventKind::TextCommit) {
                        submit(state, text);
                    } else {
                        state.input = text;
                    }
                }
            }
            SEND => {
                let text = state.input.clone();
                submit(state, text);
            }
            STOP => {
                state.cancel_requested = true;
                state.dirty = true;
            }
            CLEAR => {
                state.reset_requested = true;
                state.dirty = true;
            }
            AUTH_ACTION => {
                if state.status.starts_with("Signed in") {
                    state.logout_requested = true;
                } else {
                    state.login_requested = true;
                }
                state.dirty = true;
            }
            ALLOW => {
                state.decision = Some(true);
                state.pending = None;
                state.dirty = true;
            }
            DENY => {
                state.decision = Some(false);
                state.pending = None;
                state.dirty = true;
            }
            _ => {}
        }
    }
}

/// Record a prompt for submission, ignoring blank input.
fn submit(state: &mut SharedState, text: String) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    state.submit = Some(trimmed.to_string());
    state.input.clear();
    state.dirty = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{new_shared, Pending};

    fn state_with(f: impl FnOnce(&mut SharedState)) -> SharedState {
        let mut s = SharedState::new();
        f(&mut s);
        s
    }

    fn click(id: &str) -> PanelEvent {
        PanelEvent {
            panel_id: PANEL_ID.into(),
            control_id: id.into(),
            kind: PanelEventKind::Click,
            value: PanelValue::None,
        }
    }

    fn ids(snapshot: &PanelSnapshot) -> Vec<String> {
        snapshot
            .controls
            .iter()
            .map(|c| match c {
                PanelControl::Text { id, .. }
                | PanelControl::Button { id, .. }
                | PanelControl::ButtonRow { id, .. }
                | PanelControl::TextInput { id, .. } => id.clone(),
                PanelControl::TextArea(a) => a.id.clone(),
                PanelControl::Group(g) => g.id.clone(),
                _ => String::new(),
            })
            .collect()
    }

    fn action_row(snapshot: &PanelSnapshot) -> Vec<PanelButton> {
        snapshot
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::ButtonRow { id, buttons } if id == ACTIONS => Some(buttons.clone()),
                _ => None,
            })
            .expect("action row")
    }

    fn gate() -> Pending {
        Pending {
            call_id: "t1".into(),
            name: "run_command".into(),
            payload: "load 1crn".into(),
        }
    }

    #[test]
    fn panel_docks_at_the_bottom() {
        let panel = ChatPanel::new(new_shared());
        assert_eq!(panel.descriptor().placement, PanelPlacement::Bottom);
    }

    #[test]
    fn approval_group_appears_only_when_a_call_is_pending() {
        let quiet = build_snapshot(&state_with(|_| {}));
        assert!(!ids(&quiet).contains(&APPROVAL.to_string()));

        let gated = build_snapshot(&state_with(|s| s.pending = Some(gate())));
        assert!(ids(&gated).contains(&APPROVAL.to_string()));
    }

    #[test]
    fn nesting_stays_within_the_four_level_slint_limit() {
        // root Vec -> Group -> ButtonRow -> Button is the deepest path.
        let snapshot = build_snapshot(&state_with(|s| s.pending = Some(gate())));
        let group = snapshot
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::Group(g) => Some(g),
                _ => None,
            })
            .expect("approval group");
        for child in &group.children {
            assert!(
                !matches!(
                    child.control,
                    PanelControl::Group(_) | PanelControl::Row(_) | PanelControl::Column(_)
                ),
                "approval group nests too deeply to render"
            );
        }
    }

    #[test]
    fn send_requires_sign_in_input_and_an_idle_agent() {
        let signed_out = state_with(|s| {
            s.status = "Not signed in.".into();
            s.input = "hello".into();
        });
        assert!(!action_row(&build_snapshot(&signed_out))[0].enabled);

        let no_input = state_with(|s| s.status = "Signed in.".into());
        assert!(!action_row(&build_snapshot(&no_input))[0].enabled);

        let busy = state_with(|s| {
            s.status = "Signed in.".into();
            s.input = "hello".into();
            s.busy = true;
        });
        assert!(!action_row(&build_snapshot(&busy))[0].enabled);

        let ready = state_with(|s| {
            s.status = "Signed in.".into();
            s.input = "hello".into();
        });
        assert!(action_row(&build_snapshot(&ready))[0].enabled);
    }

    #[test]
    fn stop_is_enabled_only_while_busy() {
        assert!(!action_row(&build_snapshot(&state_with(|_| {})))[1].enabled);
        assert!(action_row(&build_snapshot(&state_with(|s| s.busy = true)))[1].enabled);
    }

    #[test]
    fn clicking_send_queues_the_draft_and_clears_the_box() {
        let mut s = state_with(|s| s.input = "show cartoon".into());
        apply_event(&mut s, &click(SEND));
        assert_eq!(s.submit.as_deref(), Some("show cartoon"));
        assert!(s.input.is_empty());
    }

    #[test]
    fn blank_prompts_are_ignored() {
        let mut s = state_with(|s| s.input = "   ".into());
        apply_event(&mut s, &click(SEND));
        assert!(s.submit.is_none());
        assert_eq!(s.input, "   ", "a blank draft should be left alone");
    }

    #[test]
    fn typing_records_a_draft_but_committing_submits() {
        let mut s = SharedState::new();
        apply_event(
            &mut s,
            &PanelEvent {
                panel_id: PANEL_ID.into(),
                control_id: PROMPT_INPUT.into(),
                kind: PanelEventKind::TextEdit,
                value: PanelValue::Text("zoom".into()),
            },
        );
        assert_eq!(s.input, "zoom");
        assert!(s.submit.is_none());

        apply_event(
            &mut s,
            &PanelEvent {
                panel_id: PANEL_ID.into(),
                control_id: PROMPT_INPUT.into(),
                kind: PanelEventKind::TextCommit,
                value: PanelValue::Text("zoom".into()),
            },
        );
        assert_eq!(s.submit.as_deref(), Some("zoom"));
    }

    #[test]
    fn allow_and_deny_record_a_decision_and_dismiss_the_gate() {
        for (control, expected) in [(ALLOW, true), (DENY, false)] {
            let mut s = state_with(|s| s.pending = Some(gate()));
            apply_event(&mut s, &click(control));
            assert_eq!(s.decision, Some(expected));
            assert!(s.pending.is_none());
        }
    }

    #[test]
    fn stop_and_clear_raise_their_request_flags() {
        let mut s = SharedState::new();
        apply_event(&mut s, &click(STOP));
        assert!(s.cancel_requested);
        apply_event(&mut s, &click(CLEAR));
        assert!(s.reset_requested);
    }

    #[test]
    fn auth_button_toggles_between_login_and_logout() {
        let mut s = state_with(|s| s.status = "Signed in.".into());
        apply_event(&mut s, &click(AUTH_ACTION));
        assert!(s.logout_requested);
        assert!(!s.login_requested);

        let mut s = state_with(|s| s.status = "Not signed in.".into());
        apply_event(&mut s, &click(AUTH_ACTION));
        assert!(s.login_requested);
    }

    #[test]
    fn highlights_cover_exactly_the_speaker_labels() {
        let s = state_with(|s| {
            s.push(Entry::User("hi".into()));
            s.push(Entry::Assistant("hello".into()));
            s.push(Entry::Note("ran a tool".into()));
        });
        let rendered = s.render();
        let highlights = speaker_highlights(&s);
        assert_eq!(highlights.len(), 3);
        for h in highlights {
            let slice = &rendered[h.start..h.end];
            assert!(
                ["You: ", "Claude: ", "\u{2022} "].contains(&slice),
                "highlight covered {slice:?} instead of a speaker label"
            );
        }
    }

    #[test]
    fn highlight_offsets_survive_multibyte_text() {
        // Byte offsets, not char offsets — a non-ASCII body must not shift the
        // following label's highlight.
        let s = state_with(|s| {
            s.push(Entry::User("\u{3b1}\u{3b2}\u{3b3}".into()));
            s.push(Entry::Assistant("ok".into()));
        });
        let rendered = s.render();
        let h = &speaker_highlights(&s)[1];
        assert_eq!(&rendered[h.start..h.end], "Claude: ");
    }

    #[test]
    fn unknown_controls_are_ignored() {
        let mut s = SharedState::new();
        apply_event(&mut s, &click("not_a_control"));
        assert!(s.submit.is_none() && !s.cancel_requested && !s.reset_requested);
    }
}
