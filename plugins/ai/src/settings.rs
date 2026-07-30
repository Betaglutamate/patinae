//! Plugin settings.
//!
//! Split registration by type: `define_plugin_settings!` only expands `bool`,
//! `i32`, and `f32`, so the string-valued settings are registered as
//! hand-written descriptors. Both halves land in the host's dynamic settings
//! registry, so `set` / `get` / `unset` work uniformly and values persist
//! through the user's `~/.patinae/patinaerc`.
//!
//! Naming: settings that apply to whichever provider is active are `ai_*`; the
//! model is per-provider (`claude_model`, `gemini_model`, `openrouter_model`)
//! because the defaults are vendor-specific, the names share no namespace, and
//! users switch back and forth — one global `ai_model` would be overwritten on
//! every switch.
//!
//! These settings are also the *only* record of which provider and model are
//! active. The panel's dropdowns write here rather than into plugin state, so
//! the picker and `set ai_provider` cannot disagree.

use patinae_plugin::define_plugin_settings;
use patinae_plugin::prelude::*;
use patinae_settings::{SettingType, SettingValue};

use crate::provider::ProviderId;

pub const PROVIDER: &str = "ai_provider";
pub const CLAUDE_MODEL: &str = "claude_model";
pub const GEMINI_MODEL: &str = "gemini_model";
pub const OPENROUTER_MODEL: &str = "openrouter_model";
pub const RECENT_MODELS: &str = "ai_recent_models";
pub const EFFORT: &str = "ai_effort";
pub const MAX_TOKENS: &str = "ai_max_tokens";
pub const AUTO_APPROVE: &str = "ai_auto_approve";
pub const ALLOW_PYTHON: &str = "ai_allow_python";
pub const CAPTURE_WIDTH: &str = "ai_capture_width";
pub const CAPTURE_HEIGHT: &str = "ai_capture_height";

/// How many models the picker pins to the top before the full list.
///
/// Small on purpose: the value of a recents list is that the models you use are
/// reachable without reading, and a long one is just the catalogue again.
pub const MAX_RECENT_MODELS: usize = 5;

pub const DEFAULT_EFFORT: &str = "medium";
pub const DEFAULT_MAX_TOKENS: i32 = 64_000;
pub const DEFAULT_CAPTURE_WIDTH: i32 = 1024;
pub const DEFAULT_CAPTURE_HEIGHT: i32 = 768;

define_plugin_settings! {
    AiSettings {
        max_tokens: i32 = DEFAULT_MAX_TOKENS, name = "ai_max_tokens";
        auto_approve: bool = false, name = "ai_auto_approve";
        allow_python: bool = true, name = "ai_allow_python";
        capture_width: i32 = DEFAULT_CAPTURE_WIDTH, name = "ai_capture_width";
        capture_height: i32 = DEFAULT_CAPTURE_HEIGHT, name = "ai_capture_height";
    }
}

/// Descriptors for the string-valued settings.
///
/// `value_hints` gives `set` named variants, so providers and effort levels are
/// discoverable by tab-completion instead of documentation.
pub fn string_descriptors() -> Vec<DynamicSettingDescriptor> {
    let hinted = |name: &str, default: &str, hints: &[&str]| DynamicSettingDescriptor {
        name: name.to_string(),
        setting_type: SettingType::String,
        default: SettingValue::String(default.to_string()),
        min: None,
        max: None,
        value_hints: hints
            .iter()
            .map(|v| ((*v).to_string(), SettingValue::String((*v).to_string())))
            .collect(),
        side_effects: vec![],
        object_overridable: false,
    };

    let provider_hints: Vec<&str> = ProviderId::ALL.iter().map(|p| p.as_str()).collect();

    vec![
        hinted(PROVIDER, ProviderId::Claude.as_str(), &provider_hints),
        hinted(CLAUDE_MODEL, ProviderId::Claude.default_model(), &[]),
        hinted(GEMINI_MODEL, ProviderId::Gemini.default_model(), &[]),
        hinted(
            OPENROUTER_MODEL,
            ProviderId::OpenRouter.default_model(),
            &[],
        ),
        hinted(
            EFFORT,
            DEFAULT_EFFORT,
            &["low", "medium", "high", "xhigh", "max"],
        ),
        // Written by the panel rather than typed, but registered like any other
        // setting so it persists through `patinaerc` and can be cleared with
        // `unset` when the list stops being useful.
        hinted(RECENT_MODELS, "", &[]),
    ]
}

/// Read a string setting from the host registry, falling back to `default`.
fn setting_string(shared: &SharedContext<'_>, name: &str, default: &str) -> String {
    match shared.setting_value(name) {
        Some(SettingValue::String(s)) if !s.trim().is_empty() => s,
        _ => default.to_string(),
    }
}

/// Which provider is currently selected.
///
/// An unrecognised value falls back to the default rather than erroring — a
/// typo in `patinaerc` should not make the agent unusable.
pub fn provider(shared: &SharedContext<'_>) -> ProviderId {
    let raw = setting_string(shared, PROVIDER, ProviderId::Claude.as_str());
    ProviderId::parse(&raw).unwrap_or(ProviderId::Claude)
}

/// Name of the model setting belonging to `id`.
pub fn model_setting(id: ProviderId) -> &'static str {
    match id {
        ProviderId::Claude => CLAUDE_MODEL,
        ProviderId::Gemini => GEMINI_MODEL,
        ProviderId::OpenRouter => OPENROUTER_MODEL,
    }
}

/// Model for the given provider.
pub fn model(shared: &SharedContext<'_>, id: ProviderId) -> String {
    setting_string(shared, model_setting(id), id.default_model())
}

/// Recently selected models, most recent first.
///
/// Stored as one comma-separated string because the settings registry has no
/// list type, and a list of model ids never contains a comma.
pub fn recent_models(shared: &SharedContext<'_>) -> Vec<String> {
    setting_string(shared, RECENT_MODELS, "")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Move `model` to the front of the recents list, capped at
/// [`MAX_RECENT_MODELS`].
///
/// Pure so the ordering rules are testable without a settings registry.
pub fn promote_recent(existing: &[String], model: &str) -> Vec<String> {
    let model = model.trim();
    if model.is_empty() {
        return existing.to_vec();
    }
    let mut out = vec![model.to_string()];
    out.extend(
        existing
            .iter()
            .filter(|m| m.as_str() != model)
            .take(MAX_RECENT_MODELS - 1)
            .cloned(),
    );
    out
}

/// Effort level, applied to whichever provider is active.
pub fn effort(shared: &SharedContext<'_>) -> String {
    setting_string(shared, EFFORT, DEFAULT_EFFORT)
}

pub fn max_tokens(shared: &SharedContext<'_>) -> u32 {
    shared.setting_int(MAX_TOKENS, DEFAULT_MAX_TOKENS).max(1) as u32
}

pub fn allow_python(shared: &SharedContext<'_>) -> bool {
    shared.setting_bool(ALLOW_PYTHON, true)
}

/// Upper bound on either capture edge.
///
/// This is the largest long edge either vendor accepts; beyond it the image is
/// downscaled server-side anyway, so the extra pixels are pure cost. It also
/// keeps the request inside what the GPU will allocate — a capture is a real
/// offscreen texture plus a full RGBA readback, so an unbounded edge is an
/// out-of-memory or device-limit failure, not just a large image.
pub const MAX_CAPTURE_EDGE: i32 = 2576;

/// Clamp one capture edge, saturating rather than truncating.
///
/// Applies to model-supplied dimensions as well as settings: `screenshot` takes
/// optional `width`/`height`, so an unclamped tool argument would walk straight
/// past the bound the settings path enforces.
pub fn clamp_capture_edge(value: u64) -> u32 {
    value.clamp(1, MAX_CAPTURE_EDGE as u64) as u32
}

/// Screenshot dimensions, clamped so a bad setting cannot request a zero-pixel
/// or absurdly large capture.
pub fn capture_size(shared: &SharedContext<'_>) -> (u32, u32) {
    let w = shared
        .setting_int(CAPTURE_WIDTH, DEFAULT_CAPTURE_WIDTH)
        .clamp(1, MAX_CAPTURE_EDGE);
    let h = shared
        .setting_int(CAPTURE_HEIGHT, DEFAULT_CAPTURE_HEIGHT)
        .clamp(1, MAX_CAPTURE_EDGE);
    (w as u32, h as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(name: &str) -> DynamicSettingDescriptor {
        string_descriptors()
            .into_iter()
            .find(|d| d.name == name)
            .unwrap_or_else(|| panic!("{name} descriptor"))
    }

    #[test]
    fn string_settings_are_registered_as_string_type() {
        for d in string_descriptors() {
            assert_eq!(d.setting_type, SettingType::String, "{}", d.name);
            assert!(matches!(d.default, SettingValue::String(_)));
        }
    }

    #[test]
    fn provider_setting_hints_every_known_provider() {
        // Derived from ProviderId::ALL, so a provider added without a hint is
        // impossible rather than merely undiscoverable.
        let hints: Vec<String> = descriptor(PROVIDER)
            .value_hints
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        let expected: Vec<&str> = ProviderId::ALL.iter().map(|p| p.as_str()).collect();
        assert_eq!(hints, expected);
        assert_eq!(hints, ["claude", "gemini", "openrouter"]);
    }

    #[test]
    fn every_provider_has_a_distinct_model_setting_registered() {
        let names: Vec<String> = string_descriptors().into_iter().map(|d| d.name).collect();
        for id in ProviderId::ALL {
            let key = model_setting(id);
            assert!(
                names.contains(&key.to_string()),
                "{key} is not registered, so `set {key}` would fail"
            );
            assert_eq!(
                descriptor(key).default,
                SettingValue::String(id.default_model().into())
            );
        }
        // Two providers sharing one model setting would silently overwrite
        // each other on every switch.
        let keys: Vec<&str> = ProviderId::ALL
            .iter()
            .map(|id| model_setting(*id))
            .collect();
        for (i, a) in keys.iter().enumerate() {
            assert!(
                !keys[i + 1..].contains(a),
                "{a} is shared between providers"
            );
        }
    }

    #[test]
    fn a_chosen_model_moves_to_the_front_of_the_recents_list() {
        let existing = vec!["a".to_string(), "b".to_string()];
        assert_eq!(promote_recent(&existing, "c"), ["c", "a", "b"]);
    }

    #[test]
    fn reselecting_a_recent_model_promotes_it_without_duplicating_it() {
        let existing = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(promote_recent(&existing, "c"), ["c", "a", "b"]);
    }

    #[test]
    fn the_recents_list_is_capped() {
        let existing: Vec<String> = (0..20).map(|i| format!("m{i}")).collect();
        assert_eq!(promote_recent(&existing, "new").len(), MAX_RECENT_MODELS);
        assert_eq!(promote_recent(&existing, "new")[0], "new");
    }

    #[test]
    fn a_blank_model_never_enters_the_recents_list() {
        let existing = vec!["a".to_string()];
        assert_eq!(promote_recent(&existing, "   "), existing);
    }

    #[test]
    fn effort_exposes_every_level_as_a_hint() {
        let hints: Vec<String> = descriptor(EFFORT)
            .value_hints
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(hints, ["low", "medium", "high", "xhigh", "max"]);
    }

    #[test]
    fn each_provider_has_its_own_model_setting_defaulted_to_its_own_model() {
        assert_eq!(
            descriptor(CLAUDE_MODEL).default,
            SettingValue::String(ProviderId::Claude.default_model().into())
        );
        assert_eq!(
            descriptor(GEMINI_MODEL).default,
            SettingValue::String(ProviderId::Gemini.default_model().into())
        );
    }

    #[test]
    fn claude_defaults_to_sonnet_5_at_medium() {
        // The test above only checks the descriptor agrees with `default_model`;
        // this one pins the value itself, so a change to either is deliberate.
        assert_eq!(ProviderId::Claude.default_model(), "claude-sonnet-5");
        assert_eq!(DEFAULT_EFFORT, "medium");
    }

    #[test]
    fn capture_edges_are_clamped_in_both_directions() {
        assert_eq!(clamp_capture_edge(1024), 1024);
        assert_eq!(clamp_capture_edge(0), 1, "a zero-pixel capture is invalid");
        assert_eq!(clamp_capture_edge(100_000), MAX_CAPTURE_EDGE as u32);
        // Saturating, not truncating: `as u32` alone would turn this into 0.
        assert_eq!(
            clamp_capture_edge(u32::MAX as u64 + 1),
            MAX_CAPTURE_EDGE as u32
        );
        assert_eq!(clamp_capture_edge(u64::MAX), MAX_CAPTURE_EDGE as u32);
    }

    #[test]
    fn macro_settings_carry_the_documented_defaults() {
        let s = AiSettings::default();
        assert_eq!(s.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(!s.auto_approve, "approval gating must default to on");
        assert!(s.allow_python);
    }

    #[test]
    fn every_macro_descriptor_is_numeric_or_bool() {
        // Guards the split: a string setting added to the macro would silently
        // fail to expand, so keep the two registration paths honest.
        for d in AiSettings::descriptors() {
            assert!(
                matches!(d.setting_type, SettingType::Int | SettingType::Bool),
                "{} is not expressible via define_plugin_settings!",
                d.name
            );
        }
    }

    #[test]
    fn shared_settings_use_provider_neutral_names() {
        // These apply to whichever provider is active, so they must not be
        // named after one of them.
        for name in [PROVIDER, EFFORT, MAX_TOKENS, AUTO_APPROVE, ALLOW_PYTHON] {
            assert!(name.starts_with("ai_"), "{name} should be provider-neutral");
        }
    }
}
