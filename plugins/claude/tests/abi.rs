//! ABI compatibility with the host loader.
//!
//! The host rejects a plugin whose ABI version or SDK version string differs
//! from its own (`crates/patinae-plugin-host/src/loader.rs`), and the SDK
//! version is compared at *patch* level — so a plugin built against 0.4.4 will
//! not load into a 0.4.5 host. That failure only shows up at runtime as a
//! silently missing plugin, which is exactly the kind of thing worth catching
//! in CI instead.
//!
//! This links the plugin as an rlib and validates the declaration the cdylib
//! exports, so it exercises the real symbol without needing `dlopen` or a GUI.

use claude_plugin::PATINAE_PLUGIN_DECLARATION as DECL;
use patinae_plugin::ffi;
use patinae_plugin_host::validate_declaration_versions;

#[test]
fn declaration_matches_the_host_abi_and_sdk_versions() {
    // SAFETY: the declaration's sdk_version is built with `AbiStr::from_static`
    // from the SDK's own `&'static str`, so the pointer is valid for `len`
    // initialized bytes for the whole program lifetime.
    let bytes = unsafe { DECL.sdk_version.as_bytes_checked(ffi::MAX_ABI_STRING_LEN) }
        .expect("sdk version is a readable abi string");
    let sdk = std::str::from_utf8(bytes).expect("sdk version is valid utf-8");

    validate_declaration_versions(DECL.abi_version, sdk)
        .expect("plugin declaration is incompatible with the host it was built against");
}

#[test]
fn declaration_exposes_the_callbacks_the_host_requires() {
    // The loader rejects a declaration missing either of these.
    assert!(DECL.register.is_some(), "host requires a register callback");
    assert!(
        DECL.capabilities & ffi::CAPABILITY_REGISTRATION != 0,
        "host requires the REGISTRATION capability bit"
    );
}

#[test]
fn declares_every_surface_the_plugin_actually_uses() {
    // Commands, a docked panel, dynamic settings, and a polling message
    // handler. If one of these bits is dropped, the corresponding surface is
    // silently ignored at load time rather than erroring.
    for (bit, name) in [
        (ffi::CAPABILITY_COMMANDS, "COMMANDS"),
        (ffi::CAPABILITY_PANELS, "PANELS"),
        (ffi::CAPABILITY_SETTINGS, "SETTINGS"),
        (ffi::CAPABILITY_MESSAGE_RUNTIME, "MESSAGE_RUNTIME"),
    ] {
        assert!(
            DECL.capabilities & bit != 0,
            "declaration is missing the {name} capability"
        );
    }
}

#[test]
fn declares_no_capability_the_host_does_not_know() {
    // The loader rejects unknown capability bits outright.
    assert_eq!(
        DECL.capabilities & !ffi::KNOWN_CAPABILITIES,
        0,
        "declaration sets a capability bit the host does not recognise"
    );
}
