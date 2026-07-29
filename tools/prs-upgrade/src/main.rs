// Rust guideline compliant 2026-02-21

use std::env;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use flate2::read::GzDecoder;
use patinae_scene::Session;
use patinae_session::prs::{
    load_prs_document, save_prs, PrsDocument, PRS_FORMAT_VERSION, PRS_PRODUCER_VERSION,
};
use patinae_settings::groups::{CartoonOverrides, CartoonSettings, Settings};
use patinae_settings::ObjectOverrides;
use rmpv::Value;
use serde::Serialize;

const HELP: &str = "\
Upgrade legacy Patinae PRS sessions to the current format.

Usage:
  prs-upgrade <input.prs> <output.prs>

The output path must not exist. The input file is never modified.
Supported legacy producers: Patinae v0.4.0 through v0.4.2.
";

// Session stored `settings` as its seventh positional field before the PRS v2
// envelope was introduced. Changing this index would target unrelated data.
const SESSION_REGISTRY_INDEX: usize = 0;
const SESSION_SETTINGS_INDEX: usize = 6;

// v0.4.0-v0.4.2 Settings had 15 positional groups. v0.4.3 inserted renderer
// at index 3 and object at index 7, shifting all later fields.
const LEGACY_SETTINGS_TO_CURRENT: [usize; 15] =
    [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const LEGACY_SETTINGS_CARTOON_INDEX: usize = 6;

// v0.4.5 inserted nucleic_ladder at index 8 and appended six dimensions to
// CartoonSettings. Existing values keep their original semantic positions.
const LEGACY_CARTOON_TO_CURRENT: [usize; 16] =
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16];

// v0.4.3 inserted ObjectSettingOverrides before the nine existing groups.
const LEGACY_OVERRIDES_TO_CURRENT: [usize; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const LEGACY_OVERRIDES_CARTOON_INDEX: usize = 0;

// ObjectRegistrySnapshot positional layout is stable across v0.4.0-v0.4.5.
// Molecule and map snapshots keep optional ObjectOverrides at these indices.
const REGISTRY_MOLECULES_INDEX: usize = 0;
const REGISTRY_MAPS_INDEX: usize = 2;
const MOLECULE_OVERRIDES_INDEX: usize = 3;
const MAP_OVERRIDES_INDEX: usize = 8;

#[derive(Debug)]
enum SourceFormat {
    CurrentRawSession,
    LegacyRawSession,
    LegacyNamedSession,
    LegacyDocument {
        format_version: Option<u64>,
        producer_version: Option<String>,
    },
}

impl fmt::Display for SourceFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CurrentRawSession => formatter.write_str("current raw Session"),
            Self::LegacyRawSession => formatter.write_str("legacy raw positional Session"),
            Self::LegacyNamedSession => formatter.write_str("legacy raw named Session"),
            Self::LegacyDocument {
                format_version,
                producer_version,
            } => {
                write!(
                    formatter,
                    "legacy PRS document (format {}, producer {})",
                    format_version
                        .map(|version| format!("v{version}"))
                        .unwrap_or_else(|| "unknown".to_string()),
                    producer_version.as_deref().unwrap_or("unknown"),
                )
            }
        }
    }
}

#[derive(Debug)]
struct UpgradeReport {
    source: SourceFormat,
    object_count: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("prs-upgrade: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let Some((input, output)) = parse_arguments()? else {
        return Ok(());
    };

    let report = upgrade_path(&input, &output)?;
    println!("Upgraded {} -> {}", input.display(), output.display());
    println!("Source: {}", report.source);
    println!("Objects: {}", report.object_count);
    println!("Output: PRS format v{PRS_FORMAT_VERSION}, Patinae {PRS_PRODUCER_VERSION}");
    Ok(())
}

fn parse_arguments() -> Result<Option<(PathBuf, PathBuf)>> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    let Some(first) = arguments.next() else {
        print!("{HELP}");
        return Ok(None);
    };

    if first == "--help" || first == "-h" {
        print!("{HELP}");
        return Ok(None);
    }
    if first == "--version" || first == "-V" {
        println!("prs-upgrade {PRS_PRODUCER_VERSION}");
        return Ok(None);
    }

    let output = arguments
        .next()
        .context("missing output path; run with --help for usage")?;
    ensure!(
        arguments.next().is_none(),
        "too many arguments; run with --help for usage"
    );
    Ok(Some((PathBuf::from(first), PathBuf::from(output))))
}

fn upgrade_path(input: &Path, output: &Path) -> Result<UpgradeReport> {
    ensure!(
        input
            .extension()
            .is_some_and(|extension| extension == "prs"),
        "input path must have a .prs extension: {}",
        input.display()
    );
    ensure!(
        output
            .extension()
            .is_some_and(|extension| extension == "prs"),
        "output path must have a .prs extension: {}",
        output.display()
    );

    let payload = read_prs_payload(input)?;
    if rmp_serde::from_slice::<PrsDocument>(&payload).is_ok() {
        bail!(
            "{} already uses a PRS document readable by this Patinae version",
            input.display()
        );
    }

    let (session, source) = match rmp_serde::from_slice::<Session>(&payload) {
        Ok(session) => (session, SourceFormat::CurrentRawSession),
        Err(current_error) => {
            let root = decode_value(&payload).with_context(|| {
                format!(
                    "{} is neither a current PRS document nor a supported legacy MessagePack value; current decoder: {current_error}",
                    input.display()
                )
            })?;
            let (mut session_value, source) = extract_session(root)?;
            migrate_session(&mut session_value)?;
            let migrated_payload = encode_value(&session_value)?;
            let session = rmp_serde::from_slice::<Session>(&migrated_payload).with_context(|| {
                format!(
                    "legacy migration produced a Session that the current decoder rejected; original decoder: {current_error}"
                )
            })?;
            (session, source)
        }
    };

    write_verified_session(&session, output)?;
    Ok(UpgradeReport {
        source,
        object_count: session.registry.len(),
    })
}

fn read_prs_payload(path: &Path) -> Result<Vec<u8>> {
    let file =
        File::open(path).with_context(|| format!("failed to open input PRS {}", path.display()))?;
    let mut decoder = GzDecoder::new(file);
    let mut payload = Vec::new();
    decoder
        .read_to_end(&mut payload)
        .with_context(|| format!("failed to decompress input PRS {}", path.display()))?;
    Ok(payload)
}

fn write_verified_session(session: &Session, output: &Path) -> Result<()> {
    let claim = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .with_context(|| {
            format!(
                "output already exists or cannot be created: {}",
                output.display()
            )
        })?;
    drop(claim);

    let result = (|| -> Result<()> {
        save_prs(session, output)
            .with_context(|| format!("failed to write upgraded PRS {}", output.display()))?;
        let document = load_prs_document(output)
            .with_context(|| format!("failed to verify upgraded PRS {}", output.display()))?;
        ensure!(
            document.prs_format_version == PRS_FORMAT_VERSION,
            "verification returned PRS format v{}, expected v{}",
            document.prs_format_version,
            PRS_FORMAT_VERSION
        );
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(output);
    }
    result
}

fn extract_session(root: Value) -> Result<(Value, SourceFormat)> {
    match root {
        Value::Array(_) => Ok((root, SourceFormat::LegacyRawSession)),
        Value::Map(mut fields) => {
            if let Some(session) = remove_map_value(&mut fields, "session") {
                let format_version =
                    map_value(&fields, "prs_format_version").and_then(Value::as_u64);
                let producer_version = map_value(&fields, "producer_version")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                Ok((
                    session,
                    SourceFormat::LegacyDocument {
                        format_version,
                        producer_version,
                    },
                ))
            } else if map_value(&fields, "registry").is_some()
                && map_value(&fields, "settings").is_some()
            {
                Ok((Value::Map(fields), SourceFormat::LegacyNamedSession))
            } else {
                bail!("unsupported named MessagePack root: no Session payload")
            }
        }
        other => bail!(
            "unsupported MessagePack root {}; expected a Session array or map",
            value_kind(&other)
        ),
    }
}

fn migrate_session(session: &mut Value) -> Result<()> {
    match session {
        Value::Array(_) => migrate_positional_session(session),
        Value::Map(_) => migrate_named_session(session),
        other => bail!(
            "unsupported Session encoding {}; expected an array or map",
            value_kind(other)
        ),
    }
}

fn migrate_positional_session(session: &mut Value) -> Result<()> {
    let fields = array_mut(session, "Session")?;
    ensure!(
        fields.len() == 10 || fields.len() == 11,
        "unsupported positional Session field count {}; expected 10 or 11",
        fields.len()
    );

    migrate_positional_registry(
        fields
            .get_mut(SESSION_REGISTRY_INDEX)
            .context("legacy Session has no registry field")?,
    )?;
    migrate_positional_settings(
        fields
            .get_mut(SESSION_SETTINGS_INDEX)
            .context("legacy Session has no settings field")?,
    )
}

fn migrate_positional_settings(settings: &mut Value) -> Result<()> {
    let mut legacy = take_array(settings, "legacy Settings")?;
    ensure!(
        legacy.len() == LEGACY_SETTINGS_TO_CURRENT.len(),
        "unsupported legacy Settings field count {}; expected {} for Patinae v0.4.0-v0.4.2",
        legacy.len(),
        LEGACY_SETTINGS_TO_CURRENT.len()
    );
    migrate_positional_cartoon(
        legacy
            .get_mut(LEGACY_SETTINGS_CARTOON_INDEX)
            .context("legacy Settings has no cartoon group")?,
        false,
    )?;
    *settings = merge_positional(
        legacy,
        positional_value(&Settings::default())?,
        &LEGACY_SETTINGS_TO_CURRENT,
        "Settings",
    )?;
    Ok(())
}

fn migrate_positional_object_overrides(overrides: &mut Value) -> Result<()> {
    let mut legacy = take_array(overrides, "legacy ObjectOverrides")?;
    ensure!(
        legacy.len() == LEGACY_OVERRIDES_TO_CURRENT.len(),
        "unsupported legacy ObjectOverrides field count {}; expected {}",
        legacy.len(),
        LEGACY_OVERRIDES_TO_CURRENT.len()
    );
    migrate_positional_cartoon(
        legacy
            .get_mut(LEGACY_OVERRIDES_CARTOON_INDEX)
            .context("legacy ObjectOverrides has no cartoon group")?,
        true,
    )?;
    *overrides = merge_positional(
        legacy,
        positional_value(&ObjectOverrides::default())?,
        &LEGACY_OVERRIDES_TO_CURRENT,
        "ObjectOverrides",
    )?;
    Ok(())
}

fn migrate_positional_cartoon(cartoon: &mut Value, overrides: bool) -> Result<()> {
    let legacy = take_array(cartoon, "legacy cartoon settings")?;
    ensure!(
        legacy.len() == LEGACY_CARTOON_TO_CURRENT.len(),
        "unsupported legacy cartoon field count {}; expected {}",
        legacy.len(),
        LEGACY_CARTOON_TO_CURRENT.len()
    );
    let defaults = if overrides {
        positional_value(&CartoonOverrides::default())?
    } else {
        positional_value(&CartoonSettings::default())?
    };
    *cartoon = merge_positional(
        legacy,
        defaults,
        &LEGACY_CARTOON_TO_CURRENT,
        "cartoon settings",
    )?;
    Ok(())
}

fn migrate_positional_registry(registry: &mut Value) -> Result<()> {
    let fields = array_mut(registry, "ObjectRegistrySnapshot")?;
    ensure!(
        fields.len() == 9,
        "unsupported ObjectRegistrySnapshot field count {}; expected 9",
        fields.len()
    );
    migrate_positional_snapshots(
        fields
            .get_mut(REGISTRY_MOLECULES_INDEX)
            .context("legacy registry has no molecules field")?,
        MOLECULE_OVERRIDES_INDEX,
        "molecule",
    )?;
    migrate_positional_snapshots(
        fields
            .get_mut(REGISTRY_MAPS_INDEX)
            .context("legacy registry has no maps field")?,
        MAP_OVERRIDES_INDEX,
        "map",
    )
}

fn migrate_positional_snapshots(
    snapshots: &mut Value,
    overrides_index: usize,
    label: &str,
) -> Result<()> {
    for (index, entry) in array_mut(snapshots, "registry snapshot collection")?
        .iter_mut()
        .enumerate()
    {
        let pair = array_mut(entry, "named object tuple")
            .with_context(|| format!("invalid {label} entry at index {index}"))?;
        ensure!(
            pair.len() == 2,
            "invalid {label} entry at index {index}: expected a name/data pair"
        );
        let snapshot = array_mut(&mut pair[1], "object snapshot")
            .with_context(|| format!("invalid {label} snapshot at index {index}"))?;
        let overrides = snapshot
            .get_mut(overrides_index)
            .with_context(|| format!("{label} snapshot at index {index} has no overrides field"))?;
        if !overrides.is_nil() {
            migrate_positional_object_overrides(overrides)
                .with_context(|| format!("failed to migrate {label} overrides at index {index}"))?;
        }
    }
    Ok(())
}

fn migrate_named_session(session: &mut Value) -> Result<()> {
    {
        let registry =
            map_value_mut(session, "registry").context("legacy Session has no registry field")?;
        migrate_named_registry(registry)?;
    }
    let settings =
        map_value_mut(session, "settings").context("legacy Session has no settings field")?;
    migrate_named_settings(settings)
}

fn migrate_named_settings(settings: &mut Value) -> Result<()> {
    if let Some(cartoon) = optional_map_value_mut(settings, "cartoon")? {
        migrate_named_cartoon(cartoon, false)?;
    }
    merge_named(settings, named_value(&Settings::default())?, "Settings")
}

fn migrate_named_object_overrides(overrides: &mut Value) -> Result<()> {
    if let Some(cartoon) = optional_map_value_mut(overrides, "cartoon")? {
        migrate_named_cartoon(cartoon, true)?;
    }
    merge_named(
        overrides,
        named_value(&ObjectOverrides::default())?,
        "ObjectOverrides",
    )
}

fn migrate_named_cartoon(cartoon: &mut Value, overrides: bool) -> Result<()> {
    let defaults = if overrides {
        named_value(&CartoonOverrides::default())?
    } else {
        named_value(&CartoonSettings::default())?
    };
    merge_named(cartoon, defaults, "cartoon settings")
}

fn migrate_named_registry(registry: &mut Value) -> Result<()> {
    if let Some(molecules) = optional_map_value_mut(registry, "molecules")? {
        migrate_named_snapshots(molecules, "molecule")?;
    }
    if let Some(maps) = optional_map_value_mut(registry, "maps")? {
        migrate_named_snapshots(maps, "map")?;
    }
    Ok(())
}

fn migrate_named_snapshots(snapshots: &mut Value, label: &str) -> Result<()> {
    for (index, entry) in array_mut(snapshots, "registry snapshot collection")?
        .iter_mut()
        .enumerate()
    {
        let pair = array_mut(entry, "named object tuple")
            .with_context(|| format!("invalid {label} entry at index {index}"))?;
        ensure!(
            pair.len() == 2,
            "invalid {label} entry at index {index}: expected a name/data pair"
        );
        if let Some(overrides) = optional_map_value_mut(&mut pair[1], "overrides")
            .with_context(|| format!("invalid {label} snapshot at index {index}"))?
        {
            if !overrides.is_nil() {
                migrate_named_object_overrides(overrides).with_context(|| {
                    format!("failed to migrate {label} overrides at index {index}")
                })?;
            }
        }
    }
    Ok(())
}

fn merge_positional(
    legacy: Vec<Value>,
    current_defaults: Value,
    legacy_to_current: &[usize],
    label: &str,
) -> Result<Value> {
    ensure!(
        legacy.len() == legacy_to_current.len(),
        "{label} migration mapping has {} entries for {} legacy fields",
        legacy_to_current.len(),
        legacy.len()
    );
    let mut current = match current_defaults {
        Value::Array(fields) => fields,
        other => bail!(
            "current {label} defaults encoded as {}, expected an array",
            value_kind(&other)
        ),
    };
    for (legacy_value, current_index) in legacy.into_iter().zip(legacy_to_current) {
        let field = current.get_mut(*current_index).with_context(|| {
            format!("{label} migration targets missing current field {current_index}")
        })?;
        *field = legacy_value;
    }
    Ok(Value::Array(current))
}

fn merge_named(target: &mut Value, current_defaults: Value, label: &str) -> Result<()> {
    let legacy = take_map(target, &format!("legacy {label}"))?;
    let mut current = match current_defaults {
        Value::Map(fields) => fields,
        other => bail!(
            "current {label} defaults encoded as {}, expected a map",
            value_kind(&other)
        ),
    };
    for (key, value) in legacy {
        if let Some((_, current_value)) =
            current.iter_mut().find(|(candidate, _)| candidate == &key)
        {
            *current_value = value;
        } else {
            current.push((key, value));
        }
    }
    *target = Value::Map(current);
    Ok(())
}

fn positional_value<T: Serialize>(value: &T) -> Result<Value> {
    let bytes = rmp_serde::to_vec(value).context("failed to encode current positional defaults")?;
    decode_value(&bytes)
}

fn named_value<T: Serialize>(value: &T) -> Result<Value> {
    let bytes =
        rmp_serde::to_vec_named(value).context("failed to encode current named defaults")?;
    decode_value(&bytes)
}

fn decode_value(bytes: &[u8]) -> Result<Value> {
    let mut cursor = Cursor::new(bytes);
    let value = rmpv::decode::read_value(&mut cursor).context("invalid MessagePack")?;
    ensure!(
        cursor.position() == bytes.len() as u64,
        "MessagePack contains {} trailing bytes",
        bytes.len() as u64 - cursor.position()
    );
    Ok(value)
}

fn encode_value(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, value)
        .context("failed to encode migrated MessagePack")?;
    Ok(bytes)
}

fn array_mut<'a>(value: &'a mut Value, label: &str) -> Result<&'a mut Vec<Value>> {
    match value {
        Value::Array(fields) => Ok(fields),
        other => bail!("{label} is {}, expected an array", value_kind(other)),
    }
}

fn take_array(value: &mut Value, label: &str) -> Result<Vec<Value>> {
    match std::mem::replace(value, Value::Nil) {
        Value::Array(fields) => Ok(fields),
        other => bail!("{label} is {}, expected an array", value_kind(&other)),
    }
}

fn take_map(value: &mut Value, label: &str) -> Result<Vec<(Value, Value)>> {
    match std::mem::replace(value, Value::Nil) {
        Value::Map(fields) => Ok(fields),
        other => bail!("{label} is {}, expected a map", value_kind(&other)),
    }
}

fn map_value<'a>(fields: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    fields
        .iter()
        .find(|(candidate, _)| candidate.as_str() == Some(key))
        .map(|(_, value)| value)
}

fn remove_map_value(fields: &mut Vec<(Value, Value)>, key: &str) -> Option<Value> {
    let index = fields
        .iter()
        .position(|(candidate, _)| candidate.as_str() == Some(key))?;
    Some(fields.remove(index).1)
}

fn map_value_mut<'a>(container: &'a mut Value, key: &str) -> Option<&'a mut Value> {
    let Value::Map(fields) = container else {
        return None;
    };
    fields
        .iter_mut()
        .find(|(candidate, _)| candidate.as_str() == Some(key))
        .map(|(_, value)| value)
}

fn optional_map_value_mut<'a>(
    container: &'a mut Value,
    key: &str,
) -> Result<Option<&'a mut Value>> {
    ensure!(
        matches!(container, Value::Map(_)),
        "named structure is {}, expected a map",
        value_kind(container)
    );
    Ok(map_value_mut(container, key))
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Nil => "nil",
        Value::Boolean(_) => "a boolean",
        Value::Integer(_) => "an integer",
        Value::F32(_) | Value::F64(_) => "a float",
        Value::String(_) => "a string",
        Value::Binary(_) => "binary data",
        Value::Array(_) => "an array",
        Value::Map(_) => "a map",
        Value::Ext(_, _) => "an extension",
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    use flate2::write::GzEncoder;
    use flate2::Compression;

    use super::*;

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    const NEW_CARTOON_FIELDS: [&str; 7] = [
        "nucleic_ladder",
        "oval_width",
        "oval_length",
        "rect_width",
        "rect_length",
        "loop_radius",
        "arrow_tip_scale",
    ];

    #[test]
    fn positional_v042_session_upgrades_and_preserves_settings() {
        let paths = TestPaths::new("positional");
        let mut session = Session::new();
        session.settings.cartoon.power = 3.25;
        let mut value = positional_value(&session).unwrap();
        downgrade_positional_session(&mut value).unwrap();
        write_gzip_value(&paths.input, &value).unwrap();

        let report = upgrade_path(&paths.input, &paths.output).unwrap();
        let upgraded = load_prs_document(&paths.output).unwrap();

        assert!(matches!(report.source, SourceFormat::LegacyRawSession));
        assert_eq!(upgraded.prs_format_version, PRS_FORMAT_VERSION);
        assert_eq!(upgraded.session.settings.cartoon.power, 3.25);
        assert!(upgraded.session.settings.cartoon.nucleic_ladder);
        assert_eq!(upgraded.session.settings.cartoon.oval_width, 0.0);
    }

    #[test]
    fn named_v042_document_upgrades_and_fills_new_fields() {
        let paths = TestPaths::new("named");
        let mut session = Session::new();
        session.settings.cartoon.power = 4.5;
        let mut session_value = named_value(&session).unwrap();
        downgrade_named_session(&mut session_value).unwrap();
        let document = Value::Map(vec![
            (Value::from("prs_format_version"), Value::from(2_u64)),
            (Value::from("producer"), Value::from("patinae")),
            (Value::from("producer_version"), Value::from("0.4.2")),
            (Value::from("session"), session_value),
        ]);
        write_gzip_value(&paths.input, &document).unwrap();

        let report = upgrade_path(&paths.input, &paths.output).unwrap();
        let upgraded = load_prs_document(&paths.output).unwrap();

        assert!(matches!(
            report.source,
            SourceFormat::LegacyDocument {
                format_version: Some(2),
                producer_version: Some(ref version),
            } if version == "0.4.2"
        ));
        assert_eq!(upgraded.session.settings.cartoon.power, 4.5);
        assert!(upgraded.session.settings.cartoon.nucleic_ladder);
        assert_eq!(upgraded.session.settings.object.state, 1);
    }

    #[test]
    fn positional_object_overrides_keep_existing_values() {
        let mut overrides = ObjectOverrides::default();
        overrides.cartoon.power = Some(6.0);
        overrides.stick.radius = Some(0.75);
        let mut value = positional_value(&overrides).unwrap();
        downgrade_positional_object_overrides(&mut value).unwrap();

        migrate_positional_object_overrides(&mut value).unwrap();
        let bytes = encode_value(&value).unwrap();
        let upgraded: ObjectOverrides = rmp_serde::from_slice(&bytes).unwrap();

        assert_eq!(upgraded.cartoon.power, Some(6.0));
        assert_eq!(upgraded.stick.radius, Some(0.75));
        assert_eq!(upgraded.object.state, None);
        assert_eq!(upgraded.cartoon.nucleic_ladder, None);
        assert_eq!(upgraded.cartoon.oval_width, None);
    }

    #[test]
    fn existing_output_is_never_overwritten() {
        let paths = TestPaths::new("existing_output");
        let mut value = positional_value(&Session::new()).unwrap();
        downgrade_positional_session(&mut value).unwrap();
        write_gzip_value(&paths.input, &value).unwrap();
        fs::write(&paths.output, b"keep me").unwrap();

        let error = upgrade_path(&paths.input, &paths.output).unwrap_err();

        assert!(error.to_string().contains("output already exists"));
        assert_eq!(fs::read(&paths.output).unwrap(), b"keep me");
    }

    #[test]
    fn current_document_is_rejected_without_creating_output() {
        let paths = TestPaths::new("current");
        save_prs(&Session::new(), &paths.input).unwrap();

        let error = upgrade_path(&paths.input, &paths.output).unwrap_err();

        assert!(error.to_string().contains("already uses a PRS document"));
        assert!(!paths.output.exists());
    }

    fn downgrade_positional_session(session: &mut Value) -> Result<()> {
        let fields = array_mut(session, "current Session")?;
        downgrade_positional_settings(
            fields
                .get_mut(SESSION_SETTINGS_INDEX)
                .context("current Session has no settings field")?,
        )
    }

    fn downgrade_positional_settings(settings: &mut Value) -> Result<()> {
        let mut current = take_array(settings, "current Settings")?;
        downgrade_positional_cartoon(
            current
                .get_mut(LEGACY_SETTINGS_TO_CURRENT[LEGACY_SETTINGS_CARTOON_INDEX])
                .context("current Settings has no cartoon group")?,
        )?;
        *settings = select_legacy_fields(
            &mut current,
            &LEGACY_SETTINGS_TO_CURRENT,
            "current Settings",
        )?;
        Ok(())
    }

    fn downgrade_positional_object_overrides(overrides: &mut Value) -> Result<()> {
        let mut current = take_array(overrides, "current ObjectOverrides")?;
        downgrade_positional_cartoon(
            current
                .get_mut(LEGACY_OVERRIDES_TO_CURRENT[LEGACY_OVERRIDES_CARTOON_INDEX])
                .context("current ObjectOverrides has no cartoon group")?,
        )?;
        *overrides = select_legacy_fields(
            &mut current,
            &LEGACY_OVERRIDES_TO_CURRENT,
            "current ObjectOverrides",
        )?;
        Ok(())
    }

    fn downgrade_positional_cartoon(cartoon: &mut Value) -> Result<()> {
        let mut current = take_array(cartoon, "current cartoon settings")?;
        *cartoon = select_legacy_fields(
            &mut current,
            &LEGACY_CARTOON_TO_CURRENT,
            "current cartoon settings",
        )?;
        Ok(())
    }

    fn select_legacy_fields(
        current: &mut [Value],
        legacy_to_current: &[usize],
        label: &str,
    ) -> Result<Value> {
        let mut legacy = Vec::with_capacity(legacy_to_current.len());
        for current_index in legacy_to_current {
            let field = current.get_mut(*current_index).with_context(|| {
                format!("{label} has no field at current index {current_index}")
            })?;
            legacy.push(std::mem::replace(field, Value::Nil));
        }
        Ok(Value::Array(legacy))
    }

    fn downgrade_named_session(session: &mut Value) -> Result<()> {
        let settings =
            map_value_mut(session, "settings").context("current Session has no settings field")?;
        let cartoon =
            map_value_mut(settings, "cartoon").context("current Settings has no cartoon group")?;
        remove_named_fields(cartoon, &NEW_CARTOON_FIELDS)?;
        remove_named_fields(settings, &["renderer", "object"])
    }

    fn remove_named_fields(value: &mut Value, names: &[&str]) -> Result<()> {
        let fields = match value {
            Value::Map(fields) => fields,
            other => bail!(
                "current named structure is {}, expected a map",
                value_kind(other)
            ),
        };
        fields.retain(|(key, _)| {
            key.as_str()
                .is_none_or(|candidate| !names.contains(&candidate))
        });
        Ok(())
    }

    fn write_gzip_value(path: &Path, value: &Value) -> Result<()> {
        let payload = encode_value(value)?;
        let file = File::create(path)
            .with_context(|| format!("failed to create test PRS {}", path.display()))?;
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(&payload)?;
        encoder.finish()?;
        Ok(())
    }

    struct TestPaths {
        dir: PathBuf,
        input: PathBuf,
        output: PathBuf,
    }

    impl TestPaths {
        fn new(name: &str) -> Self {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let dir = env::temp_dir().join(format!(
                "patinae_prs_upgrade_{name}_{}_{}",
                std::process::id(),
                id
            ));
            fs::create_dir_all(&dir).unwrap();
            Self {
                input: dir.join("legacy.prs"),
                output: dir.join("upgraded.prs"),
                dir,
            }
        }
    }

    impl Drop for TestPaths {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}
