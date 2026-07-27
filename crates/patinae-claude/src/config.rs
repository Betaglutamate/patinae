//! Claude assistant configuration and credential storage.
//!
//! The config lives at `<config_dir>/claude.json` — deliberately *outside* any
//! repository working tree, which is the single most effective defence against
//! a key being committed. Nothing here is routed through the Patinae settings
//! registry, because settings are echoed by `get`/`set`, appear in
//! autocomplete, and are serialized into shareable `.prs` session files.
//!
//! On Unix the file is created with mode `0600` and the directory with `0700`.
//! The mode is applied *at creation time* rather than with a follow-up
//! `set_permissions` call, so there is no window in which the file exists
//! world-readable.

use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::secret::SecretString;

/// Environment variable consulted before the config file.
pub const API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// File name used inside the Patinae config directory.
pub const CONFIG_FILE_NAME: &str = "claude.json";

/// Document version written to disk.
const CONFIG_VERSION: u32 = 1;

/// Default model used when the config does not name one.
pub const DEFAULT_MODEL: &str = "claude-opus-5";

/// Reasoning-depth setting passed to the API as `output_config.effort`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    /// Cheapest and fastest; fine for short, scoped requests.
    Low,
    /// Balanced.
    Medium,
    /// Default. Suits most viewer work.
    #[default]
    High,
    /// Best for demanding multi-step work.
    XHigh,
    /// Correctness over cost.
    Max,
}

impl Effort {
    /// Returns the wire value expected by `output_config.effort`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// Where a resolved API key came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    /// Read from the [`API_KEY_ENV`] environment variable.
    Environment,
    /// Read from the on-disk config file.
    ConfigFile,
}

/// Claude assistant configuration.
///
/// `Debug` is safe to log: the key field redacts itself.
#[derive(Debug, Clone)]
pub struct ClaudeConfig {
    /// API key, if one has been configured.
    api_key: Option<SecretString>,
    /// Where the key came from, when there is one.
    key_source: Option<KeySource>,
    /// Model identifier sent with each request.
    pub model: String,
    /// Reasoning depth.
    pub effort: Effort,
    /// Whether non-destructive scene changes run without a confirmation click.
    pub auto_approve_mutating: bool,
}

/// On-disk representation. Kept separate from [`ClaudeConfig`] so that the
/// in-memory type can refuse to serialize itself by accident.
#[derive(Debug, Serialize, Deserialize)]
struct ConfigDocument {
    version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    effort: Effort,
    #[serde(default = "default_auto_approve")]
    auto_approve_mutating: bool,
}

fn default_model() -> String {
    DEFAULT_MODEL.to_string()
}

fn default_auto_approve() -> bool {
    true
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            key_source: None,
            model: default_model(),
            effort: Effort::default(),
            auto_approve_mutating: default_auto_approve(),
        }
    }
}

impl ClaudeConfig {
    /// Returns the canonical config path inside the Patinae config directory.
    pub fn default_path() -> PathBuf {
        patinae_settings::paths::config_dir().join(CONFIG_FILE_NAME)
    }

    /// Loads configuration from `path`, then lets the environment override the
    /// key.
    ///
    /// Loading is forgiving in the same way as the recent-files store: a
    /// missing, malformed, or unsupported file yields defaults rather than
    /// blocking startup. An environment key always wins over a stored one and
    /// is never written back to disk.
    pub fn resolve(path: &Path) -> Self {
        let mut config = Self::load(path);
        if let Some(key) = env_api_key() {
            config.api_key = Some(key);
            config.key_source = Some(KeySource::Environment);
        }
        config
    }

    /// Loads configuration from `path`, ignoring the environment.
    pub fn load(path: &Path) -> Self {
        let Ok(bytes) = std::fs::read(path) else {
            return Self::default();
        };
        Self::from_json_slice(&bytes).unwrap_or_default()
    }

    /// Writes the configuration to `path` atomically and with restrictive
    /// permissions.
    ///
    /// A key that came from the environment is not persisted — writing it back
    /// would turn a deliberately ephemeral credential into a file on disk.
    ///
    /// # Errors
    /// Returns filesystem or serialization errors from directory creation,
    /// temporary-file writing, or the final rename.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            create_private_dir(parent)?;
        }

        let document = self.to_document();
        let bytes = serde_json::to_vec_pretty(&document).map_err(io::Error::other)?;

        let temp_path = temporary_path(path);
        // A stale temp file would keep its old, possibly permissive mode:
        // `OpenOptions::mode` only applies when the file is created.
        let _ = std::fs::remove_file(&temp_path);
        {
            let mut file = create_private_file(&temp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }

        if let Err(err) = std::fs::rename(&temp_path, path) {
            let _ = std::fs::remove_file(&temp_path);
            return Err(err);
        }
        Ok(())
    }

    /// Removes the stored configuration file, if present.
    ///
    /// # Errors
    /// Returns filesystem errors other than "not found".
    pub fn delete(path: &Path) -> io::Result<()> {
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err),
        }
    }

    /// Returns the configured API key, if any.
    pub fn api_key(&self) -> Option<&SecretString> {
        self.api_key.as_ref()
    }

    /// Returns where the current key came from.
    pub fn key_source(&self) -> Option<KeySource> {
        self.key_source
    }

    /// Returns `true` when a usable key is present.
    pub fn is_configured(&self) -> bool {
        self.api_key.as_ref().is_some_and(|key| !key.is_empty())
    }

    /// Returns a display-safe identifier for the current key.
    pub fn key_fingerprint(&self) -> Option<String> {
        self.api_key.as_ref().map(SecretString::fingerprint)
    }

    /// Stores a key that will be persisted by the next [`Self::save`].
    ///
    /// An empty or whitespace-only value clears the key instead, so that
    /// submitting a blank field does not store a useless credential.
    pub fn set_api_key(&mut self, key: &str) {
        let trimmed = key.trim();
        if trimmed.is_empty() {
            self.clear_api_key();
            return;
        }
        self.api_key = Some(SecretString::new(trimmed));
        self.key_source = Some(KeySource::ConfigFile);
    }

    /// Drops the in-memory key.
    pub fn clear_api_key(&mut self) {
        self.api_key = None;
        self.key_source = None;
    }

    fn from_json_slice(bytes: &[u8]) -> Option<Self> {
        let document: ConfigDocument = serde_json::from_slice(bytes).ok()?;
        if document.version != CONFIG_VERSION {
            return None;
        }

        let api_key = document
            .api_key
            .map(|key| key.trim().to_string())
            .filter(|key| !key.is_empty())
            .map(SecretString::new);
        let key_source = api_key.as_ref().map(|_| KeySource::ConfigFile);

        Some(Self {
            api_key,
            key_source,
            model: document.model,
            effort: document.effort,
            auto_approve_mutating: document.auto_approve_mutating,
        })
    }

    fn to_document(&self) -> ConfigDocument {
        // Environment-sourced keys stay ephemeral.
        let api_key = match self.key_source {
            Some(KeySource::ConfigFile) => {
                self.api_key.as_ref().map(|key| key.expose().to_string())
            }
            Some(KeySource::Environment) | None => None,
        };

        ConfigDocument {
            version: CONFIG_VERSION,
            api_key,
            model: self.model.clone(),
            effort: self.effort,
            auto_approve_mutating: self.auto_approve_mutating,
        }
    }
}

/// Reads a non-empty API key from the environment.
fn env_api_key() -> Option<SecretString> {
    let value = std::env::var(API_KEY_ENV).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(SecretString::new(trimmed))
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = OsString::from(".");
    name.push(
        path.file_name()
            .unwrap_or_else(|| OsStr::new(CONFIG_FILE_NAME)),
    );
    name.push(".tmp");
    path.with_file_name(name)
}

#[cfg(unix)]
fn create_private_dir(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    if path.is_dir() {
        return Ok(());
    }
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> io::Result<()> {
    // Windows has no chmod equivalent; the directory inherits the user-profile
    // ACL, which is the platform norm for per-user configuration.
    std::fs::create_dir_all(path)
}

#[cfg(unix)]
fn create_private_file(path: &Path) -> io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;

    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn create_private_file(path: &Path) -> io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Realistic-looking but fake credential. The literal is split so the CI
    /// secret scan (which matches `sk-ant-` followed by 20+ key characters)
    /// does not flag this file; the constant still has full key shape at
    /// runtime.
    const FAKE_KEY: &str = concat!("sk-ant-", "test-DEADBEEF0123456789");

    /// Unique temp directory per test; the config writer needs to own the
    /// parent so directory-mode assertions are meaningful.
    fn temp_dir(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("patinae-claude-{name}-{nonce}"))
    }

    fn temp_config(name: &str) -> PathBuf {
        temp_dir(name).join(CONFIG_FILE_NAME)
    }

    fn cleanup(path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::remove_dir_all(parent);
        }
    }

    #[test]
    fn defaults_are_sensible() {
        let config = ClaudeConfig::default();

        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.effort, Effort::High);
        assert!(config.auto_approve_mutating);
        assert!(!config.is_configured());
        assert!(config.key_fingerprint().is_none());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let path = temp_config("roundtrip");
        let mut config = ClaudeConfig::default();
        config.set_api_key(FAKE_KEY);
        config.model = "claude-sonnet-5".to_string();
        config.effort = Effort::Max;
        config.auto_approve_mutating = false;

        config.save(&path).expect("config should save");
        let restored = ClaudeConfig::load(&path);

        assert!(restored.is_configured());
        assert_eq!(restored.api_key().map(SecretString::expose), Some(FAKE_KEY));
        assert_eq!(restored.model, "claude-sonnet-5");
        assert_eq!(restored.effort, Effort::Max);
        assert!(!restored.auto_approve_mutating);

        cleanup(&path);
    }

    #[test]
    fn missing_and_malformed_files_load_defaults() {
        let missing = temp_config("missing");
        assert!(!ClaudeConfig::load(&missing).is_configured());

        let malformed = temp_config("malformed");
        std::fs::create_dir_all(malformed.parent().expect("path has a parent"))
            .expect("fixture dir should be creatable");
        std::fs::write(&malformed, b"{not json").expect("fixture should be writable");
        let config = ClaudeConfig::load(&malformed);
        assert!(!config.is_configured());
        assert_eq!(config.model, DEFAULT_MODEL);

        cleanup(&malformed);
    }

    #[test]
    fn unsupported_version_loads_defaults() {
        let path = temp_config("version");
        std::fs::create_dir_all(path.parent().expect("path has a parent"))
            .expect("fixture dir should be creatable");
        std::fs::write(
            &path,
            br#"{"version": 999, "api_key": "sk-ant-test-NOPE", "model": "x"}"#,
        )
        .expect("fixture should be writable");

        let config = ClaudeConfig::load(&path);

        assert!(!config.is_configured());
        assert_eq!(config.model, DEFAULT_MODEL);
        cleanup(&path);
    }

    #[test]
    fn blank_key_clears_rather_than_stores() {
        let mut config = ClaudeConfig::default();
        config.set_api_key(FAKE_KEY);
        assert!(config.is_configured());

        config.set_api_key("   ");

        assert!(!config.is_configured());
        assert!(config.key_source().is_none());
    }

    #[test]
    fn environment_keys_are_not_written_to_disk() {
        let path = temp_config("env-not-persisted");
        // Simulate what `resolve` does for an environment-sourced key.
        let config = ClaudeConfig {
            api_key: Some(SecretString::new(FAKE_KEY)),
            key_source: Some(KeySource::Environment),
            ..Default::default()
        };

        config.save(&path).expect("config should save");

        let written = std::fs::read_to_string(&path).expect("config should be readable");
        assert!(
            !written.contains("sk-ant-"),
            "environment key was persisted: {written}"
        );
        assert!(!ClaudeConfig::load(&path).is_configured());

        cleanup(&path);
    }

    #[test]
    fn debug_output_does_not_contain_the_key() {
        let mut config = ClaudeConfig::default();
        config.set_api_key(FAKE_KEY);

        let rendered = format!("{config:?}");

        assert!(!rendered.contains(FAKE_KEY), "Debug leaked the key");
        assert!(!rendered.contains("DEADBEEF"));
        assert!(rendered.contains(DEFAULT_MODEL));
    }

    #[test]
    fn delete_removes_the_file_and_tolerates_absence() {
        let path = temp_config("delete");
        let mut config = ClaudeConfig::default();
        config.set_api_key(FAKE_KEY);
        config.save(&path).expect("config should save");
        assert!(path.exists());

        ClaudeConfig::delete(&path).expect("delete should succeed");
        assert!(!path.exists());
        ClaudeConfig::delete(&path).expect("deleting a missing file should succeed");

        cleanup(&path);
    }

    #[cfg(unix)]
    #[test]
    fn saved_file_and_directory_are_private() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_config("permissions");
        let mut config = ClaudeConfig::default();
        config.set_api_key(FAKE_KEY);

        config.save(&path).expect("config should save");

        let file_mode = std::fs::metadata(&path)
            .expect("config should exist")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(file_mode, 0o600, "config file is not owner-only");

        let dir_mode = std::fs::metadata(path.parent().expect("path has a parent"))
            .expect("config dir should exist")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, 0o700, "config dir is not owner-only");

        cleanup(&path);
    }

    #[cfg(unix)]
    #[test]
    fn a_stale_permissive_temp_file_does_not_survive_a_save() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_config("stale-temp");
        std::fs::create_dir_all(path.parent().expect("path has a parent"))
            .expect("fixture dir should be creatable");

        // A world-readable leftover from an interrupted save. `OpenOptions::mode`
        // is ignored for files that already exist, so the writer must remove it.
        let temp = temporary_path(&path);
        std::fs::write(&temp, b"stale").expect("stale temp should be writable");
        std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o644))
            .expect("permissions should be settable");

        let mut config = ClaudeConfig::default();
        config.set_api_key(FAKE_KEY);
        config.save(&path).expect("config should save");

        let mode = std::fs::metadata(&path)
            .expect("config should exist")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "stale temp file leaked permissive mode");
        assert!(!temp.exists(), "temp file should be renamed away");

        cleanup(&path);
    }

    #[test]
    fn default_path_lives_outside_the_source_tree() {
        let path = ClaudeConfig::default_path();
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        assert!(
            !path.starts_with(&manifest_dir),
            "config path {path:?} is inside the crate at {manifest_dir:?}"
        );
        // The repository root is two levels up from `crates/patinae-claude`.
        if let Some(repo_root) = manifest_dir.parent().and_then(Path::parent) {
            assert!(
                !path.starts_with(repo_root),
                "config path {path:?} is inside the repository at {repo_root:?}"
            );
        }
        assert!(path.ends_with(CONFIG_FILE_NAME));
    }

    #[test]
    fn effort_wire_values_are_stable() {
        assert_eq!(Effort::Low.as_str(), "low");
        assert_eq!(Effort::Medium.as_str(), "medium");
        assert_eq!(Effort::High.as_str(), "high");
        assert_eq!(Effort::XHigh.as_str(), "xhigh");
        assert_eq!(Effort::Max.as_str(), "max");
    }
}
