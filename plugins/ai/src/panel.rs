//! The chat panel, docked on the right alongside Objects and Selections.
//!
//! Plugin panels are declarative: this returns a `PanelSnapshot` and the
//! frontend renders it, so nothing here touches Slint. Two constraints shape
//! the layout — the Slint bridge flattens nesting to four levels, and panel
//! callbacks cannot reach the viewer. User intent is therefore recorded as a
//! request flag in [`SharedState`] and acted on by the handler during `poll()`.
//!
//! # The layout
//!
//! Top to bottom: transcript, approval notice, composer, settings disclosure.
//!
//! The transcript takes the panel's leftover height and scrolls inside itself,
//! which is what pins everything below it on screen. That ordering is the whole
//! design: the composer and the approval buttons are the two things a user
//! needs to reach at any moment, so neither is ever allowed to be somewhere
//! they would have to scroll to find. The model picker is the opposite case —
//! a decision made about once a session — so it sits behind a one-line
//! disclosure rather than spending permanent space on a narrow dock.

use patinae_plugin::prelude::*;

use crate::provider::{ModelInfo, ProviderId};
use crate::settings;
use crate::state::{Shared, SharedState, ONBOARDING};

const PANEL_ID: &str = "ai_chat";

// Control ids, also used to route events back.
const STATUS: &str = "status";
const AUTH_ACTION: &str = "auth_action";
const TRANSCRIPT: &str = "transcript";
const COMPOSER: &str = "composer";
const SEND: &str = "send";
const STOP: &str = "stop";
const CLEAR: &str = "clear";
const APPROVAL: &str = "approval";
const ALLOW: &str = "allow";
const DENY: &str = "deny";
const SETTINGS_SECTION: &str = "settings_section";
const SETTINGS_GROUP: &str = "settings_group";
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
        // The right dock, not the bottom one. A conversation is tall and
        // narrow, the bottom dock is short and wide, and putting one in the
        // other is what forced the composer below the fold.
        PanelDescriptor::right(PANEL_ID, "AI")
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

/// Whether the selected model can drive the viewer.
///
/// Nothing known about the model is not a reason to block sending; an
/// unreachable catalogue must not disable the agent.
fn tool_capable(state: &SharedState, selection: &Selection) -> bool {
    state
        .model_info(&selection.model)
        .map(|m| m.tools)
        .unwrap_or(true)
}

/// The one-line summary on the settings disclosure.
///
/// Says what is answering and whether it can, so the collapsed state still
/// carries the two facts worth knowing without expanding anything.
fn summary_line(state: &SharedState, selection: &Selection) -> String {
    let model = state
        .model_info(&selection.model)
        .map(|m| m.label.clone())
        .unwrap_or_else(|| selection.model.clone());
    let auth = if state.status.starts_with("Signed in") {
        String::new()
    } else {
        format!(" — {}", state.status.trim_end_matches('.'))
    };
    format!(
        "{} · {} · {}{}",
        selection.provider.display_name(),
        model,
        selection.effort,
        auth
    )
}

/// Build the panel contents from state. Pure, so it is directly testable.
fn build_snapshot(state: &SharedState, selection: &Selection) -> PanelSnapshot {
    let signed_in = state.status.starts_with("Signed in");
    let capable = tool_capable(state, selection);

    // 1. The conversation. Stretches, so everything below it stays put.
    let mut controls = vec![PanelControl::Transcript(
        PanelTranscript::new(TRANSCRIPT, state.messages())
            .placeholder(ONBOARDING)
            .busy(state.busy),
    )];

    // 2. The approval gate, immediately above the composer — where the pointer
    //    already is, and close enough to the prompt that the answer is obvious.
    //    Absent entirely in auto-approve mode, so the panel stays quiet.
    if let Some(pending) = &state.pending {
        controls.push(PanelControl::Notice(
            PanelNotice::new(
                APPROVAL,
                PanelNoticeTone::Warn,
                format!("Run {}?", pending.name),
            )
            // Verbatim and monospaced: approving a command means reading
            // exactly the command that will run.
            .body(pending.payload.clone(), true)
            .buttons(vec![
                PanelButton::new(DENY, "Deny", "", false),
                PanelButton::new(ALLOW, "Allow", "", true),
            ]),
        ));
    }

    // 3. The composer, with its actions inside the frame.
    controls.push(PanelControl::Composer(
        PanelComposer::new(
            COMPOSER,
            state.input.clone(),
            PanelButton::new(SEND, "Send", "", true)
                .enabled(signed_in && capable && !state.busy && !state.input.trim().is_empty()),
        )
        .placeholder("Ask the agent to do something…")
        .max_rows(6)
        .hint("⏎ send · ⇧⏎ newline")
        .secondary(vec![
            PanelButton::new(STOP, "Stop", "", false).enabled(state.busy),
            PanelButton::new(CLEAR, "Clear", "", false).enabled(!state.transcript.is_empty()),
        ]),
    ));

    // 4. The settings disclosure. Collapsed it is one line saying what is
    //    answering; expanded it is the full picker.
    controls.push(PanelControl::Section {
        id: SETTINGS_SECTION.into(),
        title: summary_line(state, selection),
        open: state.settings_open,
    });
    if state.settings_open {
        controls.push(settings_group(state, selection, signed_in, capable));
    }

    PanelSnapshot::new(controls)
}

/// The expanded model picker.
fn settings_group(
    state: &SharedState,
    selection: &Selection,
    signed_in: bool,
    capable: bool,
) -> PanelControl {
    let options = model_options(
        &state.models,
        &selection.recents,
        &state.model_filter,
        &selection.model,
    );
    let shown = options.len();

    let mut children = vec![PanelControlNode::new(PanelControl::Row(PanelRow::new(
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
            .grow(1.0),
            PanelControlNode::new(PanelControl::Select {
                id: EFFORT_SELECT.into(),
                label: "Effort".into(),
                value: selection.effort.clone(),
                options: ["low", "medium", "high", "xhigh", "max"]
                    .iter()
                    .map(|e| PanelOption::new(*e, *e))
                    .collect(),
            })
            .grow(1.0),
        ],
    )))];

    // The model select gets its own line rather than a third of a row: ids like
    // `anthropic/claude-sonnet-5` do not fit in a third of a 280px dock.
    children.push(PanelControlNode::new(PanelControl::Select {
        id: MODEL_SELECT.into(),
        label: "Model".into(),
        value: selection.model.clone(),
        options,
    }));

    // Only worth its space once the catalogue is big enough to need it, which
    // in practice means OpenRouter and not the two native providers.
    if state.models.len() > FILTER_THRESHOLD {
        children.push(PanelControlNode::new(PanelControl::TextInput {
            id: MODEL_FILTER.into(),
            label: "".into(),
            value: state.model_filter.clone(),
            placeholder: "Filter — try `sonnet`, `gemini`, `free`".into(),
        }));
    }

    if let Some(mut line) = capability_line(state, selection, shown) {
        if !capable {
            line.push_str(
                "   — this model cannot call tools, so the agent cannot drive the viewer.",
            );
        }
        children.push(PanelControlNode::new(PanelControl::Text {
            id: CAPABILITIES.into(),
            text: line,
        }));
    }

    children.push(PanelControlNode::new(PanelControl::Button {
        id: AUTH_ACTION.into(),
        label: if signed_in { "Sign out" } else { "Sign in" }.into(),
        primary: !signed_in,
    }));

    PanelControl::Group(PanelGroup::new(SETTINGS_GROUP, "", children))
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
            COMPOSER => {
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
            SETTINGS_SECTION => {
                // The section reports the state it wants to be in.
                state.settings_open = match &event.value {
                    PanelValue::Bool(open) => *open,
                    _ => !state.settings_open,
                };
                state.dirty = true;
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

/// Record a prompt for submission and empty the box.
///
/// A blank draft submits nothing, but is still cleared: the composer empties
/// itself the moment Enter is pressed, and leaving whitespace behind in the
/// state would put the two out of step.
fn submit(state: &mut SharedState, text: String) {
    let trimmed = text.trim();
    if !trimmed.is_empty() {
        state.submit = Some(trimmed.to_string());
    }
    state.input.clear();
    state.dirty = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{new_shared, Entry, Pending};

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

    /// The controls inside the expanded settings disclosure.
    ///
    /// The picker only exists in the tree when the disclosure is open, so every
    /// test that inspects it has to open it first.
    fn settings_children(snapshot: &PanelSnapshot) -> Vec<PanelControl> {
        snapshot
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::Group(g) if g.id == SETTINGS_GROUP => Some(
                    g.children
                        .iter()
                        .map(|node| node.control.clone())
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .expect("settings group")
    }

    /// Every select in the expanded picker, by control id.
    fn select_options(snapshot: &PanelSnapshot, want: &str) -> Vec<PanelOption> {
        fn find(controls: &[PanelControl], want: &str) -> Option<Vec<PanelOption>> {
            controls.iter().find_map(|c| match c {
                PanelControl::Select { id, options, .. } if id == want => Some(options.clone()),
                PanelControl::Row(row) => find(
                    &row.children
                        .iter()
                        .map(|n| n.control.clone())
                        .collect::<Vec<_>>(),
                    want,
                ),
                _ => None,
            })
        }
        find(&settings_children(snapshot), want).expect("select")
    }

    /// The capability line inside the expanded picker.
    fn capability_text(snapshot: &PanelSnapshot) -> String {
        settings_children(snapshot)
            .iter()
            .find_map(|c| match c {
                PanelControl::Text { id, text } if id == CAPABILITIES => Some(text.clone()),
                _ => None,
            })
            .expect("capability line")
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
        snapshot.controls.iter().map(control_id).collect()
    }

    fn control_id(control: &PanelControl) -> String {
        match control {
            PanelControl::Text { id, .. }
            | PanelControl::Button { id, .. }
            | PanelControl::ButtonRow { id, .. }
            | PanelControl::Section { id, .. }
            | PanelControl::Select { id, .. }
            | PanelControl::Toggle { id, .. }
            | PanelControl::TextInput { id, .. } => id.clone(),
            PanelControl::Row(r) => r.id.clone(),
            PanelControl::TextArea(a) => a.id.clone(),
            PanelControl::Group(g) => g.id.clone(),
            PanelControl::Transcript(t) => t.id.clone(),
            PanelControl::Composer(c) => c.id.clone(),
            PanelControl::Notice(n) => n.id.clone(),
            _ => String::new(),
        }
    }

    fn composer(snapshot: &PanelSnapshot) -> PanelComposer {
        snapshot
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::Composer(c) => Some(c.clone()),
                _ => None,
            })
            .expect("composer")
    }

    fn transcript(snapshot: &PanelSnapshot) -> PanelTranscript {
        snapshot
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::Transcript(t) => Some(t.clone()),
                _ => None,
            })
            .expect("transcript")
    }

    /// Send first, then the secondary actions, in the order they are offered.
    fn action_row(snapshot: &PanelSnapshot) -> Vec<PanelButton> {
        let c = composer(snapshot);
        let mut buttons = vec![c.send];
        buttons.extend(c.secondary);
        buttons
    }

    fn gate() -> Pending {
        Pending {
            call_id: "t1".into(),
            name: "run_command".into(),
            payload: "load 1crn".into(),
        }
    }

    #[test]
    fn panel_docks_on_the_right() {
        // A conversation is tall and narrow; the bottom dock is neither.
        let panel = ChatPanel::new(new_shared());
        assert_eq!(panel.descriptor().placement, PanelPlacement::Right);
    }

    #[test]
    fn the_transcript_fills_the_panel_so_the_composer_stays_put() {
        // The whole reason the layout works: everything below the transcript is
        // pinned rather than scrolled to.
        let snapshot = snapshot_of(&state_with(|_| {}));
        assert!(snapshot.fills());
        assert_eq!(control_id(&snapshot.controls[0]), TRANSCRIPT);
        assert_eq!(
            control_id(snapshot.controls.last().unwrap()),
            SETTINGS_SECTION
        );
    }

    #[test]
    fn approval_appears_only_when_a_call_is_pending_and_sits_above_the_composer() {
        let quiet = snapshot_of(&state_with(|_| {}));
        assert!(!ids(&quiet).contains(&APPROVAL.to_string()));

        // Allow/Deny must never be somewhere the user has to scroll to find.
        let gated = snapshot_of(&state_with(|s| s.pending = Some(gate())));
        let order = ids(&gated);
        let approval = order.iter().position(|id| id == APPROVAL).expect("notice");
        let composer = order
            .iter()
            .position(|id| id == COMPOSER)
            .expect("composer");
        assert!(approval < composer);
    }

    #[test]
    fn the_approval_notice_shows_the_command_verbatim_and_as_code() {
        let gated = snapshot_of(&state_with(|s| s.pending = Some(gate())));
        let notice = gated
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::Notice(n) => Some(n.clone()),
                _ => None,
            })
            .expect("notice");

        assert_eq!(notice.body, "load 1crn");
        assert!(notice.code, "a command to approve must be monospaced");
        let labels: Vec<&str> = notice.buttons.iter().map(|b| b.label.as_str()).collect();
        assert_eq!(labels, ["Deny", "Allow"]);
    }

    #[test]
    fn the_empty_transcript_offers_examples_worth_copying() {
        let snapshot = snapshot_of(&state_with(|_| {}));
        let t = transcript(&snapshot);
        assert!(t.messages.is_empty());
        assert!(t.placeholder.contains("load 1crn"));
    }

    #[test]
    fn the_transcript_reports_the_working_state() {
        // A tool loop with nothing to show for it is indistinguishable from a
        // hang, so `busy` has to reach the view.
        assert!(!transcript(&snapshot_of(&state_with(|_| {}))).busy);
        assert!(transcript(&snapshot_of(&state_with(|s| s.busy = true))).busy);
    }

    #[test]
    fn the_transcript_carries_each_entry_as_its_own_message() {
        let snapshot = snapshot_of(&state_with(|s| {
            s.push(Entry::User("hi".into()));
            s.push(Entry::Assistant("hello".into()));
        }));
        let messages = transcript(&snapshot).messages;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, PanelMessageRole::User);
        assert_eq!(messages[1].role, PanelMessageRole::Assistant);
    }

    #[test]
    fn nesting_stays_within_the_four_level_slint_limit() {
        // The deepest path in the panel is
        // root Vec -> Group -> Row -> Select, which is exactly the limit. A
        // container one level further down would silently fail to render.
        let snapshot = snapshot_of(&state_with(|s| s.settings_open = true));
        for control in settings_children(&snapshot) {
            if let PanelControl::Row(row) = control {
                for child in &row.children {
                    assert!(
                        !matches!(
                            child.control,
                            PanelControl::Group(_) | PanelControl::Row(_) | PanelControl::Column(_)
                        ),
                        "the settings group nests too deeply to render"
                    );
                }
            }
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
    fn a_blank_prompt_submits_nothing_but_still_empties_the_box() {
        // The composer clears itself the moment Enter is pressed, so leaving
        // whitespace in the state would put the two out of step.
        let mut s = state_with(|s| s.input = "   ".into());
        apply_event(&mut s, &click(SEND), &selection());
        assert!(s.submit.is_none());
        assert!(s.input.is_empty());
    }

    #[test]
    fn a_sent_prompt_is_trimmed_before_it_reaches_the_agent() {
        let mut s = state_with(|s| s.input = "  show cartoon\n ".into());
        apply_event(&mut s, &click(SEND), &selection());
        assert_eq!(s.submit.as_deref(), Some("show cartoon"));
    }

    #[test]
    fn typing_records_a_draft_but_committing_submits() {
        let mut s = SharedState::new();
        apply_event(
            &mut s,
            &PanelEvent {
                panel_id: PANEL_ID.into(),
                control_id: COMPOSER.into(),
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
                control_id: COMPOSER.into(),
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
    fn the_settings_disclosure_starts_collapsed_and_toggles() {
        // Collapsed it is one line; expanded it is the picker. Left open it
        // would cost more of a narrow dock than the conversation above it.
        let closed = snapshot_of(&state_with(|_| {}));
        assert!(!ids(&closed).contains(&SETTINGS_GROUP.to_string()));

        let mut s = SharedState::new();
        apply_event(
            &mut s,
            &PanelEvent {
                panel_id: PANEL_ID.into(),
                control_id: SETTINGS_SECTION.into(),
                kind: PanelEventKind::Toggle,
                value: PanelValue::Bool(true),
            },
            &selection(),
        );
        assert!(s.settings_open);
        assert!(ids(&snapshot_of(&s)).contains(&SETTINGS_GROUP.to_string()));
    }

    #[test]
    fn the_collapsed_disclosure_still_says_what_is_answering() {
        let state = state_with(|s| {
            s.status = "Signed in.".into();
            s.models = vec![tool_model(ProviderId::Claude.default_model(), "Sonnet 5")];
        });
        let line = snapshot_of(&state)
            .controls
            .iter()
            .find_map(|c| match c {
                PanelControl::Section { id, title, .. } if id == SETTINGS_SECTION => {
                    Some(title.clone())
                }
                _ => None,
            })
            .expect("settings section");
        assert!(line.contains("Claude"));
        assert!(line.contains("Sonnet 5"));
        assert!(line.contains(settings::DEFAULT_EFFORT));
    }

    #[test]
    fn a_signed_out_summary_says_so_without_being_expanded() {
        // Otherwise the only clue is a Send button that will not light up.
        let state = state_with(|s| s.status = "Not signed in.".into());
        let line = summary_line(&state, &selection());
        assert!(line.contains("Not signed in"), "got: {line}");
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
        let snapshot = snapshot_of(&state_with(|s| s.settings_open = true));
        let values: Vec<String> = select_options(&snapshot, PROVIDER_SELECT)
            .iter()
            .map(|o| o.value.clone())
            .collect();
        assert_eq!(values, ["claude", "gemini", "openrouter"]);
    }

    #[test]
    fn the_model_select_gets_its_own_line_rather_than_a_third_of_a_row() {
        // `anthropic/claude-sonnet-5` does not fit in a third of a 280px dock.
        let snapshot = snapshot_of(&state_with(|s| s.settings_open = true));
        let model_is_top_level = settings_children(&snapshot)
            .iter()
            .any(|c| matches!(c, PanelControl::Select { id, .. } if id == MODEL_SELECT));
        assert!(model_is_top_level);
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
        let has_filter = |snapshot: &PanelSnapshot| {
            settings_children(snapshot)
                .iter()
                .any(|c| matches!(c, PanelControl::TextInput { id, .. } if id == MODEL_FILTER))
        };

        let small = state_with(|s| {
            s.settings_open = true;
            s.models = (0..3).map(|i| tool_model(&format!("a/{i}"), "m")).collect();
        });
        assert!(!has_filter(&snapshot_of(&small)));

        let large = state_with(|s| {
            s.settings_open = true;
            s.models = (0..FILTER_THRESHOLD + 1)
                .map(|i| tool_model(&format!("a/{i}"), "m"))
                .collect();
        });
        assert!(has_filter(&snapshot_of(&large)));
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
            s.settings_open = true;
            s.models = vec![tool_model(ProviderId::Claude.default_model(), "Sonnet")];
        });
        assert!(capability_text(&snapshot_of(&state)).contains("tools ✓"));
    }

    #[test]
    fn the_capability_line_says_when_the_list_was_truncated() {
        // Otherwise fifty entries reads as the whole catalogue.
        let mut models = vec![tool_model(ProviderId::Claude.default_model(), "Sonnet")];
        models.extend((0..MAX_OPTIONS + 20).map(|i| tool_model(&format!("v/m-{i:03}"), "m")));
        let state = state_with(|s| {
            s.settings_open = true;
            s.models = models;
        });

        let line = capability_text(&snapshot_of(&state));
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
    fn the_model_row_pairs_provider_with_effort() {
        // Two short controls share a line; the model id gets its own. Splitting
        // them the other way is what makes a 280px dock unreadable.
        let snapshot = snapshot_of(&state_with(|s| s.settings_open = true));
        let row = settings_children(&snapshot)
            .into_iter()
            .find_map(|c| match c {
                PanelControl::Row(r) if r.id == MODEL_ROW => Some(r),
                _ => None,
            })
            .expect("model row");

        let ids: Vec<String> = row
            .children
            .iter()
            .map(|n| control_id(&n.control))
            .collect();
        assert_eq!(ids, [PROVIDER_SELECT, EFFORT_SELECT]);
    }
}
