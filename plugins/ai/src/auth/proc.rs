//! Subprocess helpers shared by the credential brokers.
//!
//! Both brokers shell out to a vendor CLI, so the "is it installed / did it
//! fail / did the interactive flow hang" handling lives here once.

use std::io;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use super::AuthError;

/// Run a CLI to completion and return its output.
///
/// `install_hint` is surfaced verbatim when the binary is missing, so each
/// broker can point at its own install instructions.
pub fn run(program: &str, args: &[&str], install_hint: &str) -> Result<Output, AuthError> {
    let out = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| map_spawn_error(e, install_hint))?;
    check_status(&out)?;
    Ok(out)
}

/// Run an interactive CLI (a browser sign-in) with a deadline.
///
/// Output is captured and returned rather than inherited: a GUI user has no
/// terminal, so a URL or an error has to reach them through the panel.
pub fn run_interactive(
    program: &str,
    args: &[&str],
    install_hint: &str,
    timeout: Duration,
) -> Result<String, AuthError> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| map_spawn_error(e, install_hint))?;

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(AuthError::TimedOut(timeout.as_secs()));
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(e) => return Err(AuthError::Failed(e.to_string())),
        }
    }

    let out = child
        .wait_with_output()
        .map_err(|e| AuthError::Failed(e.to_string()))?;
    check_status(&out)?;
    Ok(combined_output(&out))
}

/// Whether a binary is on `PATH`, without running a real command.
pub fn available(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn map_spawn_error(e: io::Error, install_hint: &str) -> AuthError {
    if e.kind() == io::ErrorKind::NotFound {
        AuthError::ToolMissing(install_hint.to_string())
    } else {
        AuthError::Failed(e.to_string())
    }
}

fn check_status(out: &Output) -> Result<(), AuthError> {
    if out.status.success() {
        return Ok(());
    }
    let msg = combined_output(out);
    Err(AuthError::Failed(if msg.is_empty() {
        format!("exited with {}", out.status)
    } else {
        msg
    }))
}

pub fn combined_output(out: &Output) -> String {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let mut parts: Vec<&str> = Vec::new();
    if !stdout.trim().is_empty() {
        parts.push(stdout.trim());
    }
    if !stderr.trim().is_empty() {
        parts.push(stderr.trim());
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_binary_reports_the_install_hint() {
        let e = map_spawn_error(
            io::Error::new(io::ErrorKind::NotFound, "no such file"),
            "install me",
        );
        assert!(matches!(e, AuthError::ToolMissing(h) if h == "install me"));
    }

    #[test]
    fn other_spawn_errors_are_not_reported_as_missing() {
        let e = map_spawn_error(
            io::Error::new(io::ErrorKind::PermissionDenied, "denied"),
            "install me",
        );
        assert!(matches!(e, AuthError::Failed(_)));
    }

    #[test]
    fn a_nonexistent_binary_is_not_available() {
        assert!(!available("definitely-not-a-real-binary-xyzzy"));
    }

    #[test]
    fn failing_commands_surface_their_output() {
        // `false` exits non-zero with no output, so the status is reported.
        let err = run("false", &[], "hint").unwrap_err();
        assert!(matches!(err, AuthError::Failed(_)));
    }

    #[test]
    fn successful_commands_return_stdout() {
        let out = run("echo", &["hello"], "hint").unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hello");
    }
}
