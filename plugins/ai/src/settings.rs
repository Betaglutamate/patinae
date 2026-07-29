//! Plugin settings.
//!
//! Split registration by type: `define_plugin_settings!` only expands `bool`,
//! `i32`, and `f32`, so the string-valued settings are registered as
//! hand-written descriptors. Both halves land in the host's dynamic settings
//! registry, so `set` / `get` / `unset` work uniformly and values persist
//! through the user's `~/.patinae/patinaerc`.
//!
//! Naming: settings that apply to whichever provider is active are `ai_*`; the
//! model is per-provider (`claude_model`, `gemini_model`) because the defaults
//! are vendor-specific and users switch back and forth.

use patinae_plugin::define_plugin_settings;
use patinae_plugin::prelude::*;
use patinae_settings::{SettingType, SettingValue};

use crate::provider::ProviderId;

pub const PROVIDER: &str = "ai_provider";
pub const CLAUDE_MODEL: &str = "claude_model";
pub const GEMINI_MODEL: &str = "gemini_model";
pub const EFFORT: &str = "ai_effort";
pub const MAX_TOKENS: &str = "ai_max_tokens";
pub const AUTO_APPROVE: &str = "ai_auto_approve";
pub const ALLOW_PYTHON: &str = "ai_allow_python";
pub const CAPTURE_WIDTH: &str = "ai_capture_width";
pub const CAPTURE_HEIGHT: &str = "ai_capture_height";

pub const DEFAULT_EFFORT: &str = "xhigh";
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

    vec![
        hinted(
            PROVIDER,
            ProviderId::Claude.as_str(),
            &[ProviderId::Claude.as_str(), ProviderId::Gemini.as_str()],
        ),
        hinted(CLAUDE_MODEL, ProviderId::Claude.default_model(), &[]),
        hinted(GEMINI_MODEL, ProviderId::Gemini.default_model(), &[]),
        hinted(
            EFFORT,
            DEFAULT_EFFORT,
            &["low", "medium", "high", "xhigh", "max"],
        ),
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

/// Model for the given provider.
pub fn model(shared: &SharedContext<'_>, id: ProviderId) -> String {
    let key = match id {
        ProviderId::Claude => CLAUDE_MODEL,
        ProviderId::Gemini => GEMINI_MODEL,
    };
    setting_string(shared, key, id.default_model())
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

/// Screenshot dimensions, clamped so a bad setting cannot request a zero-pixel
/// or absurdly large capture.
///
/// The upper bound is the largest long edge either vendor accepts; beyond it
/// the image is downscaled server-side anyway, so the extra pixels are pure
/// cost.
pub fn capture_size(shared: &SharedContext<'_>) -> (u32, u32) {
    const MAX_EDGE: i32 = 2576;
    let w = shared
        .setting_int(CAPTURE_WIDTH, DEFAULT_CAPTURE_WIDTH)
        .clamp(1, MAX_EDGE);
    let h = shared
        .setting_int(CAPTURE_HEIGHT, DEFAULT_CAPTURE_HEIGHT)
        .clamp(1, MAX_EDGE);
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
        let hints: Vec<String> = descriptor(PROVIDER)
            .value_hints
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(hints, ["claude", "gemini"]);
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
