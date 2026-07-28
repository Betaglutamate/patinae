#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CARGO_BIN="${CARGO:-cargo}"
CARGO_HOST_TARGET="$("$CARGO_BIN" -vV | sed -n 's/^host: //p')"
cd "$ROOT"

toml_version() {
    local file="$1"
    local section="$2"
    awk -F '"' -v section="$section" '
        $0 == "[" section "]" { in_section = 1; next }
        /^\[/ { in_section = 0 }
        in_section && /^version = / { print $2; exit }
    ' "$file"
}

check_version() {
    local label="$1"
    local actual="$2"
    if [ "$actual" != "$VERSION" ]; then
        echo "Error: $label has version ${actual:-<missing>}; expected $VERSION" >&2
        return 1
    fi
    echo "  OK    $label ($actual)"
}

check_lockfile() {
    local manifest="$1"
    local target="$2"
    local lockfile="${manifest%Cargo.toml}Cargo.lock"
    if ! "$CARGO_BIN" metadata \
        --manifest-path "$manifest" \
        --filter-platform "$target" \
        --locked \
        --format-version 1 \
        >/dev/null; then
        echo "Error: $lockfile is stale; run 'make version V=$VERSION' and commit the result." >&2
        return 1
    fi
    echo "  OK    $lockfile"
}

VERSION=$(toml_version "Cargo.toml" "workspace.package")
if [ -z "$VERSION" ]; then
    echo "Error: could not detect the workspace version from Cargo.toml" >&2
    exit 1
fi

echo "Checking release version $VERSION..."
check_version "python/Cargo.toml" "$(toml_version "python/Cargo.toml" "package")"
check_version "python/pyproject.toml" "$(toml_version "python/pyproject.toml" "project")"
check_version "web/Cargo.toml" "$(toml_version "web/Cargo.toml" "package")"
check_version "web/package.json" "$(awk -F '"' '/^[[:space:]]*"version":/ { print $4; exit }' web/package.json)"
check_version "web/package-lock.json" "$(awk -F '"' '/^[[:space:]]*"version":/ { print $4; exit }' web/package-lock.json)"
check_version "web/package-lock.json root package" "$(awk -F '"' '/^[[:space:]]*"version":/ { if (++count == 2) { print $4; exit } }' web/package-lock.json)"
check_version "Makefile" "$(awk '/^VERSION[[:space:]]*:=/ { print $3; exit }' Makefile)"
check_version "README.md badge" "$(sed -nE 's|.*badge/version-([0-9]+\.[0-9]+\.[0-9]+)-green\.svg.*|\1|p' README.md | head -1)"

echo ""
echo "Checking Cargo lockfiles..."
check_lockfile "Cargo.toml" "$CARGO_HOST_TARGET"
check_lockfile "python/Cargo.toml" "$CARGO_HOST_TARGET"
check_lockfile "web/Cargo.toml" "wasm32-unknown-unknown"

echo ""
echo "Release metadata is consistent."
