#!/usr/bin/env bash
# lib/common.sh — shared helpers for llmforge bash scripts

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Toolchain ───────────────────────────────────────────────────────
# The parent repo pins 1.85 via rust-toolchain.toml but that toolchain
# may lack a cargo binary.  We default to stable unless the user has
# set LLMFORGE_TOOLCHAIN explicitly.
LLMFORGE_TOOLCHAIN="${LLMFORGE_TOOLCHAIN:-stable}"
export RUSTUP_TOOLCHAIN="$LLMFORGE_TOOLCHAIN"

# ── Platform detection ──────────────────────────────────────────────
detect_platform() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      echo "mingw"
      ;;
    *)
      if grep -qi microsoft /proc/version 2>/dev/null; then
        echo "wsl"
      else
        echo "linux"
      fi
      ;;
  esac
}

# ── Preflight checks ───────────────────────────────────────────────
preflight() {
  if ! command -v rustup &>/dev/null; then
    echo "ERROR: rustup not found. Install from https://rustup.rs" >&2
    exit 1
  fi
  if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found. Install Rust via rustup." >&2
    exit 1
  fi
}

# ── Target directory ────────────────────────────────────────────────
TARGET_DIR="$REPO_ROOT/target"
