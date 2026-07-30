//! On-disk cache for provider model catalogues.
//!
//! The catalogue is what makes the model picker usable — capability badges,
//! filtering to models that can actually call tools, real context sizes — but
//! fetching it is a network round trip, and OpenRouter's is several hundred
//! entries. Caching it means the picker is populated the instant the panel opens
//! rather than a second later, and that it still works offline.
//!
//! The cache is advisory in both directions. A miss costs a fetch; a stale hit
//! costs a model list that is a day out of date, which for a list of model names
//! is not a real cost. Nothing here ever fails a caller: an unreadable,
//! unwritable, or corrupt cache is simply a miss.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::provider::{ModelInfo, ProviderId};

/// How long a cached catalogue is served before being refetched.
///
/// A day. Model catalogues change on the order of weeks, and the cost of being
/// briefly out of date is that a model released this morning is missing from the
/// picker — while the setting still accepts it typed by hand.
pub const TTL_SECONDS: u64 = 24 * 60 * 60;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheFile {
    /// Seconds since the Unix epoch.
    fetched_at: u64,
    models: Vec<ModelInfo>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn cache_path(id: ProviderId) -> PathBuf {
    patinae_settings::paths::config_dir().join(format!("ai-models-{}.json", id.as_str()))
}

/// Whether a cache written at `fetched_at` is still fresh at `now`.
///
/// A timestamp in the future counts as stale rather than eternally fresh: a
/// clock that jumped forward once should not poison the cache forever.
fn is_fresh(fetched_at: u64, now: u64) -> bool {
    now >= fetched_at && now - fetched_at < TTL_SECONDS
}

/// Read a fresh catalogue, or `None` if there is no usable one.
pub fn load(id: ProviderId) -> Option<Vec<ModelInfo>> {
    let raw = std::fs::read_to_string(cache_path(id)).ok()?;
    let cache: CacheFile = serde_json::from_str(&raw).ok()?;
    if !is_fresh(cache.fetched_at, now()) || cache.models.is_empty() {
        return None;
    }
    Some(cache.models)
}

/// Write a catalogue to the cache, ignoring any failure.
///
/// An empty list is never written: it would otherwise pin a transient upstream
/// failure in place for a whole day.
pub fn store(id: ProviderId, models: &[ModelInfo]) {
    if models.is_empty() {
        return;
    }
    let path = cache_path(id);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let cache = CacheFile {
        fetched_at: now(),
        models: models.to_vec(),
    };
    if let Ok(json) = serde_json::to_string(&cache) {
        let _ = std::fs::write(&path, json);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_catalogue_within_the_ttl_is_fresh() {
        let now = 1_000_000;
        assert!(is_fresh(now, now));
        assert!(is_fresh(now - TTL_SECONDS + 1, now));
    }

    #[test]
    fn a_catalogue_older_than_the_ttl_is_stale() {
        let now = 1_000_000;
        assert!(!is_fresh(now - TTL_SECONDS, now));
        assert!(!is_fresh(now - TTL_SECONDS * 10, now));
    }

    #[test]
    fn a_timestamp_from_the_future_is_stale_rather_than_eternally_fresh() {
        // A clock that jumped forward once must not poison the cache forever.
        assert!(!is_fresh(2_000_000, 1_000_000));
    }

    #[test]
    fn each_provider_gets_its_own_cache_file() {
        let paths: Vec<PathBuf> = ProviderId::ALL.iter().map(|id| cache_path(*id)).collect();
        for (i, a) in paths.iter().enumerate() {
            for b in paths.iter().skip(i + 1) {
                assert_ne!(a, b, "two providers share a cache file");
            }
        }
    }

    #[test]
    fn a_catalogue_survives_a_round_trip_through_the_cache_format() {
        let models = vec![ModelInfo::new("a/b", "A B").with_context(1024)];
        let json = serde_json::to_string(&CacheFile {
            fetched_at: now(),
            models: models.clone(),
        })
        .unwrap();
        let back: CacheFile = serde_json::from_str(&json).unwrap();
        assert_eq!(back.models, models);
    }

    #[test]
    fn a_corrupt_or_missing_cache_is_a_miss_not_a_failure() {
        // `load` swallows every error by design; these are the paths it takes.
        assert!(serde_json::from_str::<CacheFile>("{not json").is_err());
        assert!(serde_json::from_str::<CacheFile>("{}").is_err());
    }
}
