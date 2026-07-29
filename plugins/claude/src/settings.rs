//! Plugin settings.
//!
//! Split registration by type: `define_plugin_settings!` only expands `bool`,
//! `i32`, and `f32`, so the two string-valued settings are registered as
//! hand-written descriptors. Both halves land in the host's dynamic settings
//! registry, so `set` / `get` / `unset` work uniformly and values persist
//! through the user's `~/.patinae/patinaerc`.

use patinae_plugin::define_plugin_settings;
use patinae_plugin::prelude::*;
use patinae_settings::{SettingType, SettingValue};

use crate::api::types::DEFAULT_MODEL;
use crate::worker::Config;

pub const MODEL: &str = "claude_model";
pub const EFFORT: &str = "claude_effort";
pub const MAX_TOKENS: &str = "claude_max_tokens";
pub const AUTO_APPROVE: &str = "claude_auto_approve";
pub const ALLOW_PYTHON: &str = "claude_allow_python";
pub const CAPTURE_WIDTH: &str = "claude_capture_width";
pub const CAPTURE_HEIGHT: &str = "claude_capture_height";

pub const DEFAULT_EFFORT: &str = "medium";
pub const DEFAULT_MAX_TOKENS: i32 = 64_000;
pub const DEFAULT_CAPTURE_WIDTH: i32 = 1024;
pub const DEFAULT_CAPTURE_HEIGHT: i32 = 768;

define_plugin_settings! {
    ClaudeSettings {
        max_tokens: i32 = DEFAULT_MAX_TOKENS, name = "claude_max_tokens";
        auto_approve: bool = false, name = "claude_auto_approve";
        allow_python: bool = true, name = "claude_allow_python";
        capture_width: i32 = DEFAULT_CAPTURE_WIDTH, name = "claude_capture_width";
        capture_height: i32 = DEFAULT_CAPTURE_HEIGHT, name = "claude_capture_height";
    }
}

/// Descriptors for the string-valued settings.
///
/// `value_hints` gives `set claude_effort, ` named variants, so the effort
/// levels are discoverable by tab-completion instead of documentation.
pub fn string_descriptors() -> Vec<DynamicSettingDescriptor> {
    vec![
        DynamicSettingDescriptor {
            name: MODEL.to_string(),
            setting_type: SettingType::String,
            default: SettingValue::String(DEFAULT_MODEL.to_string()),
            min: None,
            max: None,
            value_hints: vec![],
            side_effects: vec![],
            object_overridable: false,
        },
        DynamicSettingDescriptor {
            name: EFFORT.to_string(),
            setting_type: SettingType::String,
            default: SettingValue::String(DEFAULT_EFFORT.to_string()),
            min: None,
            max: None,
            value_hints: ["low", "medium", "high", "xhigh", "max"]
                .into_iter()
                .map(|v| (v.to_string(), SettingValue::String(v.to_string())))
                .collect(),
            side_effects: vec![],
            object_overridable: false,
        },
    ]
}

/// Read a string setting from the host registry, falling back to `default`.
fn setting_string(shared: &SharedContext<'_>, name: &str, default: &str) -> String {
    match shared.setting_value(name) {
        Some(SettingValue::String(s)) if !s.trim().is_empty() => s,
        _ => default.to_string(),
    }
}

/// Snapshot the current settings into a worker [`Config`].
pub fn read_config(shared: &SharedContext<'_>, system_prompt: String) -> Config {
    Config {
        model: setting_string(shared, MODEL, DEFAULT_MODEL),
        effort: setting_string(shared, EFFORT, DEFAULT_EFFORT),
        max_tokens: shared.setting_int(MAX_TOKENS, DEFAULT_MAX_TOKENS).max(1) as u32,
        allow_python: shared.setting_bool(ALLOW_PYTHON, true),
        system_prompt,
    }
}

/// Upper bound on either capture edge.
///
/// This is the model's maximum long edge; beyond it the image is downscaled
/// server-side anyway, so the extra pixels are pure cost. It also keeps the
/// request inside what the GPU will allocate — a capture is a real offscreen
/// texture plus a full RGBA readback, so an unbounded edge is an out-of-memory
/// or device-limit failure, not just a large image.
pub const MAX_CAPTURE_EDGE: i32 = 2576;

/// Clamp one capture edge, saturating rather than truncating.
///
/// Applies to model-supplied dimensions as well as settings: `screenshot` takes
/// optional `width`/`height`, so an unclamped tool argument would walk straight
/// past the bound the settings path enforces.
pub fn clamp_capture_edge(value: u64) -> u32 {
    value.clamp(1, MAX_CAPTURE_EDGE as u64) as u32
}

/// Screenshot dimensions, clamped so a bad setting cannot request a
/// zero-pixel or absurdly large capture.
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

    #[test]
    fn string_settings_are_registered_as_string_type() {
        for d in string_descriptors() {
            assert_eq!(d.setting_type, SettingType::String, "{}", d.name);
            assert!(matches!(d.default, SettingValue::String(_)));
        }
    }

    #[test]
    fn effort_exposes_every_level_as_a_hint() {
        let effort = string_descriptors()
            .into_iter()
            .find(|d| d.name == EFFORT)
            .expect("effort descriptor");
        let hints: Vec<String> = effort.value_hints.into_iter().map(|(k, _)| k).collect();
        assert_eq!(hints, ["low", "medium", "high", "xhigh", "max"]);
    }

    #[test]
    fn model_defaults_to_sonnet_5() {
        let model = string_descriptors()
            .into_iter()
            .find(|d| d.name == MODEL)
            .expect("model descriptor");
        assert_eq!(model.default, SettingValue::String("claude-sonnet-5".into()));
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
        let s = ClaudeSettings::default();
        assert_eq!(s.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(!s.auto_approve, "approval gating must default to on");
        assert!(s.allow_python);
    }

    #[test]
    fn every_macro_descriptor_is_numeric_or_bool() {
        // Guards the split: a string setting added to the macro would silently
        // fail to expand, so keep the two registration paths honest.
        for d in ClaudeSettings::descriptors() {
            assert!(
                matches!(d.setting_type, SettingType::Int | SettingType::Bool),
                "{} is not expressible via define_plugin_settings!",
                d.name
            );
        }
    }
}
