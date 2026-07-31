//! Declarative plugin UI contracts.
//!
//! Plugins describe panel contents with these data types. Frontends own the
//! actual rendering, so plugin UI stays independent of egui, Slint, or any
//! other widget toolkit.

use crate::component::SharedContext;
use crate::message::MessageBus;
use serde::{Deserialize, Serialize};

/// Dock location supported by Patinae plugin panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelPlacement {
    Right,
    Bottom,
}

/// Static metadata for a plugin panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelDescriptor {
    pub id: String,
    pub title: String,
    /// Short text/icon shown in the left toolbar.
    pub icon: String,
    pub placement: PanelPlacement,
    pub default_visible: bool,
}

impl PanelDescriptor {
    pub fn right(id: impl Into<String>, title: impl Into<String>) -> Self {
        let title = title.into();
        Self {
            id: id.into(),
            icon: default_icon(&title),
            title,
            placement: PanelPlacement::Right,
            default_visible: false,
        }
    }

    pub fn bottom(id: impl Into<String>, title: impl Into<String>) -> Self {
        let title = title.into();
        Self {
            id: id.into(),
            icon: default_icon(&title),
            title,
            placement: PanelPlacement::Bottom,
            default_visible: false,
        }
    }

    pub fn icon(mut self, icon: impl Into<String>) -> Self {
        self.icon = icon.into();
        self
    }

    pub fn default_visible(mut self, visible: bool) -> Self {
        self.default_visible = visible;
        self
    }
}

fn default_icon(title: &str) -> String {
    title
        .chars()
        .find(|c| !c.is_whitespace())
        .map(|c| c.to_uppercase().to_string())
        .unwrap_or_else(|| "+".to_string())
}

/// One selectable option for segmented/dropdown-style controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelOption {
    pub label: String,
    pub value: String,
}

impl PanelOption {
    pub fn new(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
        }
    }
}

/// A compact icon button used inside toolbar-style rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelButton {
    pub id: String,
    pub label: String,
    pub icon: String,
    pub primary: bool,
    pub enabled: bool,
}

impl PanelButton {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        icon: impl Into<String>,
        primary: bool,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            icon: icon.into(),
            primary,
            enabled: true,
        }
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Semantic style for highlighted text ranges in plugin text areas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelTextStyle {
    Keyword,
    String,
    Comment,
    Number,
    Function,
    Type,
    Constant,
    Operator,
    Punctuation,
    Builtin,
}

impl PanelTextStyle {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Keyword => "keyword",
            Self::String => "string",
            Self::Comment => "comment",
            Self::Number => "number",
            Self::Function => "function",
            Self::Type => "type",
            Self::Constant => "constant",
            Self::Operator => "operator",
            Self::Punctuation => "punctuation",
            Self::Builtin => "builtin",
        }
    }
}

/// A highlighted byte range inside a plugin text area.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PanelTextHighlight {
    pub start: usize,
    pub end: usize,
    pub style: PanelTextStyle,
}

impl PanelTextHighlight {
    pub fn new(start: usize, end: usize, style: PanelTextStyle) -> Self {
        Self { start, end, style }
    }
}

/// A multi-line text surface used by scripting/editor panels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelTextArea {
    pub id: String,
    pub label: String,
    pub value: String,
    pub placeholder: String,
    pub rows: u32,
    pub read_only: bool,
    pub highlights: Vec<PanelTextHighlight>,
}

impl PanelTextArea {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        value: impl Into<String>,
        placeholder: impl Into<String>,
        rows: u32,
        read_only: bool,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            value: value.into(),
            placeholder: placeholder.into(),
            rows,
            read_only,
            highlights: Vec::new(),
        }
    }

    pub fn with_highlights(mut self, highlights: Vec<PanelTextHighlight>) -> Self {
        self.highlights = highlights;
        self
    }
}

// =============================================================================
// Conversational controls
// =============================================================================
//
// A chat is not a stack of form fields. These three describe one — a scrolling
// log of authored messages, a composer that stays put beneath it, and a card
// for the moments the user has to answer something. They render at the top
// level of a panel only, which is why none of them appear in the nested
// `PanelControlNode` vocabulary: a transcript inside a table cell is not a
// layout anyone wants.

/// Who authored a transcript message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelMessageRole {
    User,
    Assistant,
    /// Something the agent *did* rather than said — a tool call, rendered as an
    /// auditable row rather than as prose.
    Tool,
    Error,
}

/// How a message's action turned out.
///
/// Only meaningful for [`PanelMessageRole::Tool`]; everything else is
/// [`PanelMessageStatus::None`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelMessageStatus {
    None,
    Running,
    Ok,
    Denied,
    Failed,
}

/// One span of a message body.
///
/// Bodies arrive as markdown-ish text; the plugin splits them so the renderer
/// never has to parse anything. Code earns different treatment from prose —
/// monospace, its own surface — and that is the distinction worth carrying.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelMessageBlock {
    Prose(String),
    Code { language: String, text: String },
}

impl PanelMessageBlock {
    pub fn prose(text: impl Into<String>) -> Self {
        Self::Prose(text.into())
    }

    pub fn code(language: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Code {
            language: language.into(),
            text: text.into(),
        }
    }

    /// The block's text, whichever kind it is.
    pub fn text(&self) -> &str {
        match self {
            Self::Prose(text) => text,
            Self::Code { text, .. } => text,
        }
    }
}

/// One authored entry in a transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelMessage {
    /// Stable across snapshots, so the renderer can reuse the row it already
    /// built instead of rebuilding the list on every streamed token.
    pub id: String,
    pub role: PanelMessageRole,
    /// Display name of the speaker — "You", "Claude", "Gemini".
    pub author: String,
    pub blocks: Vec<PanelMessageBlock>,
    pub status: PanelMessageStatus,
    /// Secondary text: the command for a tool row, the model for a reply.
    pub detail: String,
}

impl PanelMessage {
    pub fn new(
        id: impl Into<String>,
        role: PanelMessageRole,
        author: impl Into<String>,
        blocks: Vec<PanelMessageBlock>,
    ) -> Self {
        Self {
            id: id.into(),
            role,
            author: author.into(),
            blocks,
            status: PanelMessageStatus::None,
            detail: String::new(),
        }
    }

    /// A message whose body is a single run of prose.
    pub fn text(
        id: impl Into<String>,
        role: PanelMessageRole,
        author: impl Into<String>,
        body: impl Into<String>,
    ) -> Self {
        Self::new(id, role, author, vec![PanelMessageBlock::prose(body)])
    }

    pub fn status(mut self, status: PanelMessageStatus) -> Self {
        self.status = status;
        self
    }

    pub fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }
}

/// A scrolling conversation log.
///
/// Stretches to fill whatever height the panel has left, and scrolls inside
/// itself — which is what keeps the composer beneath it on screen instead of
/// pushed below the fold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelTranscript {
    pub id: String,
    pub messages: Vec<PanelMessage>,
    /// Shown in place of the messages while the transcript is empty.
    pub placeholder: String,
    /// Whether to show a working indicator at the tail. A long tool loop with
    /// no visible sign of life is indistinguishable from a hang.
    pub busy: bool,
}

impl PanelTranscript {
    pub fn new(id: impl Into<String>, messages: Vec<PanelMessage>) -> Self {
        Self {
            id: id.into(),
            messages,
            placeholder: String::new(),
            busy: false,
        }
    }

    pub fn placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    pub fn busy(mut self, busy: bool) -> Self {
        self.busy = busy;
        self
    }
}

/// A multi-line prompt box with its actions attached.
///
/// The send button lives inside the frame rather than in a separate row: the
/// text and the act of sending it are one object, and pairing them means the
/// composer cannot be on screen while its button is not.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelComposer {
    pub id: String,
    pub value: String,
    pub placeholder: String,
    /// Lines to grow to before the box starts scrolling instead.
    pub max_rows: u32,
    /// The primary action, rendered inside the frame.
    pub send: PanelButton,
    /// Buttons rendered alongside it — stop, clear.
    pub secondary: Vec<PanelButton>,
    /// Keyboard hint shown under the box, e.g. "⏎ send · ⇧⏎ newline".
    pub hint: String,
}

impl PanelComposer {
    pub fn new(id: impl Into<String>, value: impl Into<String>, send: PanelButton) -> Self {
        Self {
            id: id.into(),
            value: value.into(),
            placeholder: String::new(),
            max_rows: 6,
            send,
            secondary: Vec::new(),
            hint: String::new(),
        }
    }

    pub fn placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    pub fn max_rows(mut self, rows: u32) -> Self {
        self.max_rows = rows.max(1);
        self
    }

    pub fn secondary(mut self, buttons: Vec<PanelButton>) -> Self {
        self.secondary = buttons;
        self
    }

    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = hint.into();
        self
    }
}

/// How loudly a notice presents itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelNoticeTone {
    Info,
    Warn,
    Danger,
}

/// An accent-ruled card asking for attention, and usually for an answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelNotice {
    pub id: String,
    pub tone: PanelNoticeTone,
    pub title: String,
    pub body: String,
    /// Render the body monospaced. Approval prompts show the exact command that
    /// would run, and a proportional font is the wrong typeface for deciding
    /// whether to run something.
    pub code: bool,
    pub buttons: Vec<PanelButton>,
}

impl PanelNotice {
    pub fn new(id: impl Into<String>, tone: PanelNoticeTone, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tone,
            title: title.into(),
            body: String::new(),
            code: false,
            buttons: Vec::new(),
        }
    }

    pub fn body(mut self, body: impl Into<String>, code: bool) -> Self {
        self.body = body.into();
        self.code = code;
        self
    }

    pub fn buttons(mut self, buttons: Vec<PanelButton>) -> Self {
        self.buttons = buttons;
        self
    }
}

/// A child control inside a generic panel layout container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelControlNode {
    pub control: PanelControl,
    pub grow: f32,
}

impl PanelControlNode {
    pub fn new(control: PanelControl) -> Self {
        Self { control, grow: 0.0 }
    }

    pub fn grow(mut self, grow: f32) -> Self {
        self.grow = grow;
        self
    }
}

/// A horizontal group of plugin controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelRow {
    pub id: String,
    pub children: Vec<PanelControlNode>,
    pub gap: f32,
}

impl PanelRow {
    pub fn new(id: impl Into<String>, children: Vec<PanelControlNode>) -> Self {
        Self {
            id: id.into(),
            children,
            gap: 8.0,
        }
    }

    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }
}

/// A vertical group of plugin controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelColumn {
    pub id: String,
    pub children: Vec<PanelControlNode>,
    pub gap: f32,
}

impl PanelColumn {
    pub fn new(id: impl Into<String>, children: Vec<PanelControlNode>) -> Self {
        Self {
            id: id.into(),
            children,
            gap: 4.0,
        }
    }

    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }
}

/// A titled visual group of plugin controls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelGroup {
    pub id: String,
    pub title: String,
    pub open: bool,
    pub children: Vec<PanelControlNode>,
    pub gap: f32,
}

impl PanelGroup {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        children: Vec<PanelControlNode>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            open: true,
            children,
            gap: 8.0,
        }
    }

    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }
}

/// Controls supported by the v1 declarative renderer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanelControl {
    Text {
        id: String,
        text: String,
    },
    /// A bold title with a muted description beneath — a compact header block
    /// that pairs well with an action button to its right.
    TitleDesc {
        id: String,
        title: String,
        desc: String,
    },
    Heading {
        id: String,
        text: String,
    },
    Section {
        id: String,
        title: String,
        open: bool,
    },
    Button {
        id: String,
        label: String,
        primary: bool,
    },
    ButtonRow {
        id: String,
        buttons: Vec<PanelButton>,
    },
    Toggle {
        id: String,
        label: String,
        value: bool,
    },
    Slider {
        id: String,
        label: String,
        value: f32,
        min: f32,
        max: f32,
        step: f32,
    },
    Number {
        id: String,
        label: String,
        value: f32,
        min: f32,
        max: f32,
        step: f32,
    },
    Select {
        id: String,
        label: String,
        value: String,
        options: Vec<PanelOption>,
    },
    TextInput {
        id: String,
        label: String,
        value: String,
        placeholder: String,
    },
    TextArea(PanelTextArea),
    /// A scrolling conversation log. Absorbs the panel's leftover height.
    Transcript(PanelTranscript),
    /// A multi-line prompt box with its send action inside the frame.
    Composer(PanelComposer),
    /// An accent-ruled card asking for attention.
    Notice(PanelNotice),
    Row(PanelRow),
    Column(PanelColumn),
    Group(PanelGroup),
    Image {
        id: String,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    },
    Spacer {
        id: String,
        height: f32,
    },
}

impl PanelControl {
    /// Whether this control absorbs the panel's leftover vertical space.
    ///
    /// Every other control declares a fixed height and the panel scrolls as a
    /// whole. A transcript instead takes what is left and scrolls internally,
    /// which is the difference between a composer that stays put and one that
    /// slides off the bottom of the dock.
    pub fn stretches(&self) -> bool {
        matches!(self, Self::Transcript(_))
    }
}

/// A complete render snapshot for one panel.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PanelSnapshot {
    pub controls: Vec<PanelControl>,
}

impl PanelSnapshot {
    pub fn new(controls: Vec<PanelControl>) -> Self {
        Self { controls }
    }

    /// Whether the panel lays out to its full height instead of scrolling.
    ///
    /// True as soon as one control stretches. The frontend uses this to choose
    /// between a scrolling stack and a filled layout, so panels that predate
    /// the conversational controls keep their existing behaviour untouched.
    pub fn fills(&self) -> bool {
        self.controls.iter().any(PanelControl::stretches)
    }
}

/// Runtime value sent from the frontend to a plugin panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanelValue {
    None,
    Bool(bool),
    Number(f32),
    Text(String),
}

impl PanelValue {
    pub fn as_command_value(&self) -> String {
        match self {
            Self::None => String::new(),
            Self::Bool(v) => {
                if *v {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
            Self::Number(v) => format!("{v:.6}")
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
            Self::Text(v) => v.clone(),
        }
    }
}

/// Frontend event kind emitted by rendered plugin controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelEventKind {
    Click,
    Toggle,
    NumberChange,
    Select,
    TextEdit,
    TextAreaEdit,
    TextCommit,
}

impl PanelEventKind {
    pub fn refreshes_snapshot(self) -> bool {
        !matches!(self, Self::TextEdit)
    }
}

/// An event emitted by the rendered panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelEvent {
    pub panel_id: String,
    pub control_id: String,
    pub kind: PanelEventKind,
    pub value: PanelValue,
}

/// High-level actions returned by plugin panels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanelAction {
    ExecuteCommand { command: String, silent: bool },
    SetSetting { name: String, value: PanelValue },
    Custom { topic: String, payload: Vec<u8> },
}

/// Runtime host inputs requested by a plugin panel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PanelRuntimeRequirements {
    bits: u64,
}

impl PanelRuntimeRequirements {
    /// No extra host runtime inputs are required.
    pub const NONE: Self = Self { bits: 0 };

    /// The panel needs a full serialized session.
    pub const FULL_SESSION: Self = Self { bits: 1 << 0 };

    /// Builds requirements from raw ABI bits.
    pub const fn from_bits(bits: u64) -> Self {
        Self { bits }
    }

    /// Returns the raw ABI bitset.
    pub const fn bits(self) -> u64 {
        self.bits
    }

    /// Returns true when all requested bits are present.
    pub const fn contains(self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }

    /// Returns the union of two requirement sets.
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Returns true when no extra runtime inputs are required.
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }
}

/// A UI panel implemented by a plugin.
pub trait PluginPanel: Send {
    fn descriptor(&self) -> PanelDescriptor;

    fn runtime_requirements(&self) -> PanelRuntimeRequirements {
        PanelRuntimeRequirements::FULL_SESSION
    }

    fn snapshot(&mut self, ctx: &SharedContext<'_>) -> PanelSnapshot;

    fn handle_event(
        &mut self,
        _event: PanelEvent,
        _ctx: &SharedContext<'_>,
        _bus: &mut MessageBus,
    ) -> Vec<PanelAction> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript() -> PanelControl {
        PanelControl::Transcript(PanelTranscript::new(
            "log",
            vec![PanelMessage::text(
                "m1",
                PanelMessageRole::User,
                "You",
                "hi",
            )],
        ))
    }

    #[test]
    fn only_a_transcript_absorbs_leftover_height() {
        assert!(transcript().stretches());
        assert!(!PanelControl::Text {
            id: "t".into(),
            text: "x".into(),
        }
        .stretches());
        assert!(!PanelControl::TextArea(PanelTextArea::new("a", "", "", "", 4, false)).stretches());
    }

    #[test]
    fn a_panel_fills_exactly_when_one_of_its_controls_stretches() {
        // Panels that predate the conversational controls must keep scrolling
        // as a whole, so this has to stay false for all of them.
        let form = PanelSnapshot::new(vec![PanelControl::Text {
            id: "t".into(),
            text: "x".into(),
        }]);
        assert!(!form.fills());
        assert!(!PanelSnapshot::default().fills());

        let chat = PanelSnapshot::new(vec![
            transcript(),
            PanelControl::Composer(PanelComposer::new(
                "c",
                "",
                PanelButton::new("send", "Send", "", true),
            )),
        ]);
        assert!(chat.fills());
    }

    #[test]
    fn a_message_block_yields_its_text_whichever_kind_it_is() {
        assert_eq!(PanelMessageBlock::prose("hello").text(), "hello");
        assert_eq!(
            PanelMessageBlock::code("python", "cmd.zoom()").text(),
            "cmd.zoom()"
        );
    }

    #[test]
    fn messages_default_to_no_status_and_carry_one_when_given() {
        let plain = PanelMessage::text("m", PanelMessageRole::Assistant, "Claude", "hi");
        assert_eq!(plain.status, PanelMessageStatus::None);

        let tool = PanelMessage::text("t", PanelMessageRole::Tool, "run_command", "")
            .status(PanelMessageStatus::Running)
            .detail("load 1crn");
        assert_eq!(tool.status, PanelMessageStatus::Running);
        assert_eq!(tool.detail, "load 1crn");
    }

    #[test]
    fn a_composer_always_has_at_least_one_row_to_type_into() {
        let composer =
            PanelComposer::new("c", "", PanelButton::new("send", "Send", "", true)).max_rows(0);
        assert_eq!(composer.max_rows, 1);
    }

    #[test]
    fn text_edit_is_the_only_non_refreshing_panel_event() {
        assert!(!PanelEventKind::TextEdit.refreshes_snapshot());
        assert!(PanelEventKind::Click.refreshes_snapshot());
        assert!(PanelEventKind::Toggle.refreshes_snapshot());
        assert!(PanelEventKind::NumberChange.refreshes_snapshot());
        assert!(PanelEventKind::Select.refreshes_snapshot());
        assert!(PanelEventKind::TextAreaEdit.refreshes_snapshot());
        assert!(PanelEventKind::TextCommit.refreshes_snapshot());
    }
}
