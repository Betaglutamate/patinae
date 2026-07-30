//! The chat panel, docked at the bottom alongside the Python script editor.
//!
//! Plugin panels are declarative: this returns a `PanelSnapshot` and the
//! frontend renders it, so nothing here touches Slint. Two constraints shape
//! the layout — the Slint bridge flattens nesting to four levels, and panel
//! callbacks cannot reach the viewer. User intent is therefore recorded as a
//! request flag in [`SharedState`] and acted on by the handler during `poll()`.

use patinae_plugin::prelude::*;

use crate::provider::{ModelInfo, ProviderId};
use crate::settings;
use crate::state::{Entry, Shared, SharedState};

const PANEL_ID: &str = "ai_chat";

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
const MODEL_ROW: &str = "model_row";
const PROVIDER_SELECT: &str = "provider";
const MODEL_SELECT: &str = "model";
const EFFORT_SELECT: &str = "effort";
const MODEL_FILTER: &str = "model_filter";
const CAPABILITIES: &str = "capabilities";

/// Catalogue size above which the filter box appears.
///
/// Claude and Gemini publish a handful of models and need no filter; OpenRouter
/// publishes several hundred and is unusable without one. Keying on the size
/// rather than on the provider means the box appears exactly when it earns its
/// space.
const FILTER_THRESHOLD: usize = 12;

/// Most models offered in the dropdown at once.
///
/// A native select listing four hundred entries is not a picker, it is a wall.
/// The filter box and the recents list are how you reach anything past this.
const MAX_OPTIONS: usize = 50;

pub struct ChatPanel {
    state: Shared,
}

impl ChatPanel {
    pub fn new(state: Shared) -> Self {
        Self { state }
    }
}

/// The settings-backed half of what the panel renders.
///
/// Read fresh from the host registry on every snapshot rather than mirrored into
/// [`SharedState`]. That keeps one source of truth: `set ai_provider, gemini` at
/// the command line and the dropdown above are the same act, and neither can
/// leave the other showing something stale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub provider: ProviderId,
    pub model: String,
    pub effort: String,
    pub recents: Vec<String>,
}

impl Selection {
    fn read(ctx: &SharedContext<'_>) -> Self {
        let provider = settings::provider(ctx);
        Self {
            provider,
            model: settings::model(ctx, provider),
            effort: settings::effort(ctx),
            recents: settings::recent_models(ctx),
        }
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
        let prefix = state.prefix(entry);
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

// =============================================================================
// The model picker
// =============================================================================

/// Whether a model matches the filter box.
///
/// Matches on id and label together so both "sonnet" and "Anthropic" find the
/// same entry, and splits the query on whitespace so "claude free" narrows
/// rather than finding nothing.
fn matches_filter(model: &ModelInfo, filter: &str) -> bool {
    let filter = filter.trim().to_ascii_lowercase();
    if filter.is_empty() {
        return true;
    }
    let haystack = format!("{} {}", model.id, model.label).to_ascii_lowercase();
    filter
        .split_whitespace()
        .all(|term| haystack.contains(term))
}

/// Build the model dropdown's options.
///
/// Three rules, in order of how much they matter:
///
/// 1. **Models that cannot call tools are excluded.** The agent's entire job is
///    driving the viewer through tools; offering a model that cannot is offering
///    a trap. They are reachable by typing the id into `openrouter_model`.
/// 2. **Recently used models come first.** This is what makes a
///    several-hundred-entry catalogue feel small: the two or three you actually
///    use are always at the top.
/// 3. **The current selection is always present**, even when the filter or the
///    tool rule would exclude it — a dropdown that cannot show what is selected
///    would silently reset it on the next interaction.
pub fn model_options(
    models: &[ModelInfo],
    recents: &[String],
    filter: &str,
    current: &str,
) -> Vec<PanelOption> {
    let label_for = |id: &str| -> String {
        models
            .iter()
            .find(|m| m.id == id)
            .map(|m| m.label.clone())
            .unwrap_or_else(|| id.to_string())
    };

    let mut ids: Vec<String> = Vec::new();
    let push = |id: String, ids: &mut Vec<String>| {
        if !ids.contains(&id) {
            ids.push(id);
        }
    };

    push(current.to_string(), &mut ids);
    for id in recents {
        if models.iter().any(|m| m.id == *id) {
            push(id.clone(), &mut ids);
        }
    }
    for model in models {
        if ids.len() >= MAX_OPTIONS {
            break;
        }
        if model.tools && matches_filter(model, filter) {
            push(model.id.clone(), &mut ids);
        }
    }

    ids.into_iter()
        .map(|id| PanelOption::new(label_for(&id), id))
        .collect()
}

/// The line under the picker describing what is selected.
fn capability_line(state: &SharedState, selection: &Selection, shown: usize) -> Option<String> {
    let info = state.model_info(&selection.model)?;
    let mut line = info.badges();

    // Say when the list is truncated, or a user who filters for nothing and
    // sees fifty entries will reasonably conclude that is the whole catalogue.
    let offered = state.models.iter().filter(|m| m.tools).count();
    if shown < offered {
        line.push_str(&format!(
            "   ({shown} of {offered} tool-capable models shown — type to filter)"
        ));
    }
    Some(line)
}

impl PluginPanel for ChatPanel {
    fn descriptor(&self) -> PanelDescriptor {
        PanelDescriptor::bottom(PANEL_ID, "AI")
            .icon("AI")
            .default_visible(false)
    }

    fn runtime_requirements(&self) -> PanelRuntimeRequirements {
        // The panel renders plugin-owned state only, so there is no reason to
        // pay for a serialized session snapshot every frame.
        PanelRuntimeRequirements::NONE
    }

    fn snapshot(&mut self, ctx: &SharedContext<'_>) -> PanelSnapshot {
        match self.state.lock() {
            Ok(mut state) => {
                // Rendering is the signal that someone can actually see the
                // picker, and therefore that the catalogue is worth fetching.
                state.panel_shown = true;
                build_snapshot(&state, &Selection::read(ctx))
            }
            Err(_) => PanelSnapshot::new(vec![PanelControl::Text {
                id: STATUS.into(),
                text: "AI panel state is unavailable.".into(),
            }]),
        }
    }

    fn handle_event(
        &mut self,
        event: PanelEvent,
        ctx: &SharedContext<'_>,
        _bus: &mut MessageBus,
    ) -> Vec<PanelAction> {
        let selection = Selection::read(ctx);
        let Ok(mut state) = self.state.lock() else {
            return Vec::new();
        };
        // Everything else is deferred to the handler's poll, which is the only
        // place with viewer access. Settings are the exception: they go back to
        // the host directly, which is what keeps the dropdowns and the `set`
        // command reading from the same place.
        apply_event(&mut state, &event, &selection)
    }
}

/// Build the panel contents from state. Pure, so it is directly testable.
fn build_snapshot(state: &SharedState, selection: &Selection) -> PanelSnapshot {
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

        // The model bar sits directly above the prompt, where the decision is
        // actually made, rather than in a settings dialog somewhere else.
        let options = model_options(
            &state.models,
            &selection.recents,
            &state.model_filter,
            &selection.model,
        );
        controls.push(PanelControl::Row(PanelRow::new(
            MODEL_ROW,
            vec![
                PanelControlNode::new(PanelControl::Select {
                    id: PROVIDER_SELECT.into(),
                    label: "Provider".into(),
                    value: selection.provider.as_str().to_string(),
                    options: ProviderId::ALL
                        .iter()
                        .map(|p| PanelOption::new(p.display_name(), p.as_str()))
                        .collect(),
                })
                .grow(0.8),
                PanelControlNode::new(PanelControl::Select {
                    id: MODEL_SELECT.into(),
                    label: "Model".into(),
                    value: selection.model.clone(),
                    options,
                })
                .grow(2.0),
                PanelControlNode::new(PanelControl::Select {
                    id: EFFORT_SELECT.into(),
                    label: "Effort".into(),
                    value: selection.effort.clone(),
                    options: ["low", "medium", "high", "xhigh", "max"]
                        .iter()
                        .map(|e| PanelOption::new(*e, *e))
                        .collect(),
                })
                .grow(0.7),
            ],
        )));

        // Only worth its space once the catalogue is big enough to need it,
        // which in practice means OpenRouter and not the two native providers.
        if state.models.len() > FILTER_THRESHOLD {
            controls.push(PanelControl::TextInput {
                id: MODEL_FILTER.into(),
                label: "".into(),
                value: state.model_filter.clone(),
                placeholder: "Filter models — try `sonnet`, `gemini`, `free`".into(),
            });
        }

        let capabilities = capability_line(state, selection, model_option_count(&controls));
        let tool_capable = state
            .model_info(&selection.model)
            .map(|m| m.tools)
            // Nothing known about the model is not a reason to block sending;
            // an unreachable catalogue must not disable the agent.
            .unwrap_or(true);
        if let Some(mut line) = capabilities {
            if !tool_capable {
                line.push_str(
                    "   — this model cannot call tools, so the agent cannot drive the viewer.",
                );
            }
            controls.push(PanelControl::Text {
                id: CAPABILITIES.into(),
                text: line,
            });
        }

        controls.push(PanelControl::TextInput {
            id: PROMPT_INPUT.into(),
            label: "".into(),
            value: state.input.clone(),
            placeholder: "Ask the agent to do something…".into(),
        });
        controls.push(PanelControl::ButtonRow {
            id: ACTIONS.into(),
            buttons: vec![
                PanelButton::new(SEND, "Send", "", true).enabled(
                    signed_in && tool_capable && !state.busy && !state.input.trim().is_empty(),
                ),
                PanelButton::new(STOP, "Stop", "", false).enabled(state.busy),
                PanelButton::new(CLEAR, "Clear", "", false).enabled(!state.transcript.is_empty()),
            ],
        });

        PanelSnapshot::new(controls)
    }
}

/// How many options the model dropdown ended up with.
fn model_option_count(controls: &[PanelControl]) -> usize {
    controls
        .iter()
        .find_map(|c| match c {
            PanelControl::Row(row) => row.children.iter().find_map(|node| match &node.control {
                PanelControl::Select { id, options, .. } if id == MODEL_SELECT => {
                    Some(options.len())
                }
                _ => None,
            }),
            _ => None,
        })
        .unwrap_or(0)
}

fn set_setting(name: &str, value: impl Into<String>) -> PanelAction {
    PanelAction::SetSetting {
        name: name.to_string(),
        value: PanelValue::Text(value.into()),
    }
}

/// Apply one panel event to state, returning any settings it changes.
///
/// Pure, so both halves are directly testable.
fn apply_event(
    state: &mut SharedState,
    event: &PanelEvent,
    selection: &Selection,
) -> Vec<PanelAction> {
    {
        match event.control_id.as_str() {
            PROVIDER_SELECT => {
                let Some(raw) = text_value(&event.value) else {
                    return Vec::new();
                };
                let Some(provider) = ProviderId::parse(&raw) else {
                    return Vec::new();
                };
                if provider == selection.provider {
                    return Vec::new();
                }
                // The catalogue belongs to the provider we just left, so drop it
                // rather than briefly offering one vendor's models under
                // another's name. The handler refetches on the next poll.
                state.models.clear();
                state.models_provider = None;
                state.model_filter.clear();
                state.assistant_label = provider.display_name().to_string();
                state.dirty = true;
                return vec![set_setting(settings::PROVIDER, provider.as_str())];
            }
            MODEL_SELECT => {
                let Some(model) = text_value(&event.value).filter(|m| !m.trim().is_empty()) else {
                    return Vec::new();
                };
                if model == selection.model {
                    return Vec::new();
                }
                state.dirty = true;
                // Recents are what keep a several-hundred-entry catalogue
                // usable, so every deliberate pick feeds them.
                return vec![
                    set_setting(settings::model_setting(selection.provider), model.clone()),
                    set_setting(
                        settings::RECENT_MODELS,
                        settings::promote_recent(&selection.recents, &model).join(","),
                    ),
                ];
            }
            EFFORT_SELECT => {
                let Some(effort) = text_value(&event.value) else {
                    return Vec::new();
                };
                state.dirty = true;
                return vec![set_setting(settings::EFFORT, effort)];
            }
            MODEL_FILTER => {
                if let Some(text) = text_value(&event.value) {
                    state.model_filter = text;
                    state.dirty = true;
                }
                return Vec::new();
            }
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
    Vec::new()
}

/// Text carried by a panel event, whichever value shape it arrived in.
fn text_value(value: &PanelValue) -> Option<String> {
    match value {
        PanelValue::Text(t) => Some(t.clone()),
        _ => None,
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

    /// The default selection: Claude at its own default model and effort.
    fn selection() -> Selection {
        Selection {
            provider: ProviderId::Claude,
            model: ProviderId::Claude.default_model().to_string(),
            effort: settings::DEFAULT_EFFORT.to_string(),
            recents: Vec::new(),
        }
    }

    fn snapshot_of(state: &SharedState) -> PanelSnapshot {
        build_snapshot(state, &selection())
    }

    fn click(id: &str) -> PanelEvent {
        PanelEvent {
            panel_id: PANEL_ID.into(),
            control_id: id.into(),
            kind: PanelEventKind::Click,
            value: PanelValue::None,
        }
    }

    fn choose(id: &str, value: &str) -> PanelEvent {
        PanelEvent {
            panel_id: PANEL_ID.into(),
            control_id: id.into(),
            kind: PanelEventKind::Select,
            value: PanelValue::Text(value.into()),
        }
    }

    fn tool_model(id: &str, label: &str) -> ModelInfo {
        ModelInfo::new(id, label)
    }

    fn toolless_model(id: &str, label: &str) -> ModelInfo {
        let mut m = ModelInfo::new(id, label);
        m.tools = false;
        m
    }

    fn settings_set(actions: &[PanelAction]) -> Vec<(String, String)> {
        actions
            .iter()
            .filter_map(|a| match a {
                PanelAction::SetSetting {
                    name,
                    value: PanelValue::Text(v),
                } => Some((name.clone(), v.clone())),
                _ => None,
            })
            .collect()
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
        let quiet = snapshot_of(&state_with(|_| {}));
        assert!(!ids(&quiet).contains(&APPROVAL.to_string()));

        let gated = snapshot_of(&state_with(|s| s.pending = Some(gate())));
        assert!(ids(&gated).contains(&APPROVAL.to_string()));
    }

    #[test]
    fn nesting_stays_within_the_four_level_slint_limit() {
        // root Vec -> Group -> ButtonRow -> Button is the deepest path.
        let snapshot = snapshot_of(&state_with(|s| s.pending = Some(gate())));
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
        assert!(!action_row(&snapshot_of(&signed_out))[0].enabled);

        let no_input = state_with(|s| s.status = "Signed in.".into());
        assert!(!action_row(&snapshot_of(&no_input))[0].enabled);

        let busy = state_with(|s| {
            s.status = "Signed in.".into();
            s.input = "hello".into();
            s.busy = true;
        });
        assert!(!action_row(&snapshot_of(&busy))[0].enabled);

        let ready = state_with(|s| {
            s.status = "Signed in.".into();
            s.input = "hello".into();
        });
        assert!(action_row(&snapshot_of(&ready))[0].enabled);
    }

    #[test]
    fn stop_is_enabled_only_while_busy() {
        assert!(!action_row(&snapshot_of(&state_with(|_| {})))[1].enabled);
        assert!(action_row(&snapshot_of(&state_with(|s| s.busy = true)))[1].enabled);
    }

    #[test]
    fn clicking_send_queues_the_draft_and_clears_the_box() {
        let mut s = state_with(|s| s.input = "show cartoon".into());
        apply_event(&mut s, &click(SEND), &selection());
        assert_eq!(s.submit.as_deref(), Some("show cartoon"));
        assert!(s.input.is_empty());
    }

    #[test]
    fn blank_prompts_are_ignored() {
        let mut s = state_with(|s| s.input = "   ".into());
        apply_event(&mut s, &click(SEND), &selection());
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
            &selection(),
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
            &selection(),
        );
        assert_eq!(s.submit.as_deref(), Some("zoom"));
    }

    #[test]
    fn allow_and_deny_record_a_decision_and_dismiss_the_gate() {
        for (control, expected) in [(ALLOW, true), (DENY, false)] {
            let mut s = state_with(|s| s.pending = Some(gate()));
            apply_event(&mut s, &click(control), &selection());
            assert_eq!(s.decision, Some(expected));
            assert!(s.pending.is_none());
        }
    }

    #[test]
    fn stop_and_clear_raise_their_request_flags() {
        let mut s = SharedState::new();
        apply_event(&mut s, &click(STOP), &selection());
        assert!(s.cancel_requested);
        apply_event(&mut s, &click(CLEAR), &selection());
        assert!(s.reset_requested);
    }

    #[test]
    fn auth_button_toggles_between_login_and_logout() {
        let mut s = state_with(|s| s.status = "Signed in.".into());
        apply_event(&mut s, &click(AUTH_ACTION), &selection());
        assert!(s.logout_requested);
        assert!(!s.login_requested);

        let mut s = state_with(|s| s.status = "Not signed in.".into());
        apply_event(&mut s, &click(AUTH_ACTION), &selection());
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
        apply_event(&mut s, &click("not_a_control"), &selection());
        assert!(s.submit.is_none() && !s.cancel_requested && !s.reset_requested);
    }

    // --- the model switcher ------------------------------------------------

    #[test]
    fn the_model_bar_offers_every_provider() {
        let snapshot = snapshot_of(&state_with(|_| {}));
        let providers = snapshot
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::Row(row) => row.children.iter().find_map(|n| match &n.control {
                    PanelControl::Select { id, options, .. } if id == PROVIDER_SELECT => {
                        Some(options.clone())
                    }
                    _ => None,
                }),
                _ => None,
            })
            .expect("provider select");

        let values: Vec<&str> = providers.iter().map(|o| o.value.as_str()).collect();
        assert_eq!(values, ["claude", "gemini", "openrouter"]);
    }

    #[test]
    fn switching_provider_writes_the_setting_the_command_line_also_writes() {
        // One source of truth: the dropdown and `set ai_provider` are the same
        // act, so neither can leave the other stale.
        let mut s = SharedState::new();
        let actions = apply_event(&mut s, &choose(PROVIDER_SELECT, "openrouter"), &selection());
        assert_eq!(
            settings_set(&actions),
            [(settings::PROVIDER.to_string(), "openrouter".to_string())]
        );
    }

    #[test]
    fn switching_provider_discards_the_previous_catalogue_and_filter() {
        // Otherwise one vendor's models are briefly listed under another's name.
        let mut s = state_with(|s| {
            s.models = vec![tool_model("claude-sonnet-5", "Sonnet")];
            s.models_provider = Some(ProviderId::Claude);
            s.model_filter = "sonnet".into();
        });
        apply_event(&mut s, &choose(PROVIDER_SELECT, "gemini"), &selection());
        assert!(s.models.is_empty());
        assert_eq!(s.models_provider, None);
        assert!(s.model_filter.is_empty());
        assert_eq!(s.assistant_label, "Gemini");
    }

    #[test]
    fn reselecting_the_current_provider_changes_nothing() {
        let mut s = state_with(|s| {
            s.models = vec![tool_model("claude-sonnet-5", "Sonnet")];
            s.models_provider = Some(ProviderId::Claude);
        });
        let actions = apply_event(&mut s, &choose(PROVIDER_SELECT, "claude"), &selection());
        assert!(actions.is_empty());
        assert!(
            !s.models.is_empty(),
            "the catalogue must not be thrown away"
        );
    }

    #[test]
    fn choosing_a_model_writes_it_to_that_providers_own_setting() {
        let mut s = SharedState::new();
        let picked = Selection {
            provider: ProviderId::OpenRouter,
            model: "anthropic/claude-sonnet-5".into(),
            ..selection()
        };
        let actions = apply_event(
            &mut s,
            &choose(MODEL_SELECT, "google/gemini-2.5-pro"),
            &picked,
        );
        let set = settings_set(&actions);
        assert_eq!(
            set[0],
            (
                settings::OPENROUTER_MODEL.to_string(),
                "google/gemini-2.5-pro".to_string()
            )
        );
    }

    #[test]
    fn choosing_a_model_promotes_it_in_the_recents_list() {
        let mut s = SharedState::new();
        let picked = Selection {
            recents: vec!["a/one".into(), "b/two".into()],
            ..selection()
        };
        let actions = apply_event(&mut s, &choose(MODEL_SELECT, "c/three"), &picked);
        let recents = settings_set(&actions)
            .into_iter()
            .find(|(name, _)| name == settings::RECENT_MODELS)
            .expect("recents were not updated");
        assert_eq!(recents.1, "c/three,a/one,b/two");
    }

    #[test]
    fn changing_effort_writes_the_shared_setting() {
        let mut s = SharedState::new();
        let actions = apply_event(&mut s, &choose(EFFORT_SELECT, "high"), &selection());
        assert_eq!(
            settings_set(&actions),
            [(settings::EFFORT.to_string(), "high".to_string())]
        );
    }

    #[test]
    fn the_filter_box_is_ui_state_and_never_becomes_a_setting() {
        // Nobody wants their filter box restored at startup.
        let mut s = SharedState::new();
        let actions = apply_event(&mut s, &choose(MODEL_FILTER, "sonnet"), &selection());
        assert!(actions.is_empty());
        assert_eq!(s.model_filter, "sonnet");
    }

    #[test]
    fn the_filter_box_appears_only_for_a_catalogue_big_enough_to_need_it() {
        let small = state_with(|s| {
            s.models = (0..3).map(|i| tool_model(&format!("a/{i}"), "m")).collect();
        });
        assert!(!ids(&snapshot_of(&small)).contains(&MODEL_FILTER.to_string()));

        let large = state_with(|s| {
            s.models = (0..FILTER_THRESHOLD + 1)
                .map(|i| tool_model(&format!("a/{i}"), "m"))
                .collect();
        });
        assert!(ids(&snapshot_of(&large)).contains(&MODEL_FILTER.to_string()));
    }

    #[test]
    fn models_that_cannot_call_tools_are_kept_out_of_the_picker() {
        // The agent's whole job is driving the viewer through tools; offering
        // one that cannot is offering a trap.
        let models = vec![
            tool_model("a/capable", "Capable"),
            toolless_model("b/inert", "Inert"),
        ];
        let options = model_options(&models, &[], "", "a/capable");
        let values: Vec<&str> = options.iter().map(|o| o.value.as_str()).collect();
        assert_eq!(values, ["a/capable"]);
    }

    #[test]
    fn the_current_selection_is_always_offered_even_when_it_would_be_excluded() {
        // A dropdown that cannot show what is selected silently resets it.
        let models = vec![
            toolless_model("b/inert", "Inert"),
            tool_model("a/capable", "Capable"),
        ];
        let options = model_options(&models, &[], "nothing-matches-this", "b/inert");
        assert_eq!(options[0].value, "b/inert");
        assert_eq!(options[0].label, "Inert", "it keeps its catalogue label");
    }

    #[test]
    fn a_hand_typed_model_missing_from_the_catalogue_still_appears() {
        let options = model_options(&[], &[], "", "x-ai/grok-5");
        assert_eq!(options.len(), 1);
        assert_eq!(options[0].value, "x-ai/grok-5");
        assert_eq!(options[0].label, "x-ai/grok-5", "falls back to the id");
    }

    #[test]
    fn recent_models_sort_above_the_rest_of_the_catalogue() {
        let models = vec![
            tool_model("a/alpha", "Alpha"),
            tool_model("m/middle", "Middle"),
            tool_model("z/zulu", "Zulu"),
        ];
        let options = model_options(&models, &["z/zulu".into()], "", "m/middle");
        let values: Vec<&str> = options.iter().map(|o| o.value.as_str()).collect();
        assert_eq!(values, ["m/middle", "z/zulu", "a/alpha"]);
    }

    #[test]
    fn a_recent_model_no_longer_in_the_catalogue_is_not_offered() {
        let models = vec![tool_model("a/alpha", "Alpha")];
        let options = model_options(&models, &["gone/model".into()], "", "a/alpha");
        let values: Vec<&str> = options.iter().map(|o| o.value.as_str()).collect();
        assert_eq!(values, ["a/alpha"]);
    }

    #[test]
    fn the_filter_matches_on_id_and_label_and_narrows_across_terms() {
        let models = vec![
            tool_model("anthropic/claude-sonnet-5", "Anthropic: Claude Sonnet 5"),
            tool_model("google/gemini-2.5-pro", "Google: Gemini 2.5 Pro"),
            tool_model("anthropic/claude-opus-5", "Anthropic: Claude Opus 5"),
        ];

        let by_id = model_options(&models, &[], "sonnet", "anthropic/claude-sonnet-5");
        assert_eq!(by_id.len(), 1);

        // Matching the label, not the id.
        let by_vendor = model_options(&models, &[], "Google", "google/gemini-2.5-pro");
        assert_eq!(by_vendor.len(), 1);

        // Terms narrow rather than conflict.
        let two_terms = model_options(&models, &[], "anthropic opus", "anthropic/claude-opus-5");
        assert_eq!(two_terms.len(), 1);
        assert_eq!(two_terms[0].value, "anthropic/claude-opus-5");

        // Case and stray whitespace must not matter.
        assert_eq!(
            model_options(&models, &[], "  SONNET  ", "anthropic/claude-sonnet-5").len(),
            1
        );
    }

    #[test]
    fn a_huge_catalogue_is_capped_rather_than_dumped_into_the_dropdown() {
        let models: Vec<ModelInfo> = (0..400)
            .map(|i| tool_model(&format!("v/model-{i:03}"), "m"))
            .collect();
        let options = model_options(&models, &[], "", "v/model-000");
        assert_eq!(options.len(), MAX_OPTIONS);
    }

    #[test]
    fn the_capability_line_describes_the_selected_model() {
        let state = state_with(|s| {
            s.models = vec![tool_model(ProviderId::Claude.default_model(), "Sonnet")];
        });
        let snapshot = snapshot_of(&state);
        let line = snapshot
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::Text { id, text } if id == CAPABILITIES => Some(text.clone()),
                _ => None,
            })
            .expect("capability line");
        assert!(line.contains("tools ✓"));
    }

    #[test]
    fn the_capability_line_says_when_the_list_was_truncated() {
        // Otherwise fifty entries reads as the whole catalogue.
        let mut models = vec![tool_model(ProviderId::Claude.default_model(), "Sonnet")];
        models.extend((0..MAX_OPTIONS + 20).map(|i| tool_model(&format!("v/m-{i:03}"), "m")));
        let state = state_with(|s| s.models = models);

        let snapshot = snapshot_of(&state);
        let line = snapshot
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::Text { id, text } if id == CAPABILITIES => Some(text.clone()),
                _ => None,
            })
            .expect("capability line");
        assert!(line.contains("type to filter"), "got: {line}");
    }

    #[test]
    fn send_is_disabled_when_the_selected_model_cannot_call_tools() {
        let state = state_with(|s| {
            s.status = "Signed in.".into();
            s.input = "hello".into();
            s.models = vec![toolless_model(ProviderId::Claude.default_model(), "Inert")];
        });
        assert!(!action_row(&snapshot_of(&state))[0].enabled);
    }

    #[test]
    fn an_unreachable_catalogue_does_not_disable_the_agent() {
        // Knowing nothing about a model is not a reason to block sending.
        let state = state_with(|s| {
            s.status = "Signed in.".into();
            s.input = "hello".into();
        });
        assert!(state.models.is_empty());
        assert!(action_row(&snapshot_of(&state))[0].enabled);
    }

    #[test]
    fn the_model_row_stays_within_the_slint_nesting_limit() {
        // root Vec -> Row -> Select is the deepest path here.
        let snapshot = snapshot_of(&state_with(|_| {}));
        let row = snapshot
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::Row(r) if r.id == MODEL_ROW => Some(r),
                _ => None,
            })
            .expect("model row");
        for child in &row.children {
            assert!(
                !matches!(
                    child.control,
                    PanelControl::Group(_) | PanelControl::Row(_) | PanelControl::Column(_)
                ),
                "the model row nests too deeply to render"
            );
        }
    }
}
