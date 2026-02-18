#!/usr/bin/env bash
# lib/common.sh — shared helpers for rust-deeplearning bash scripts

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Platform detection ───────────────────────────────────────────────
detect_platform() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
      echo "mingw"
      ;;
    Darwin*)
      echo "macos"
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

# ── Preflight checks ────────────────────────────────────────────────
preflight() {
  if [ "${SKIP_PREFLIGHT:-}" = "1" ]; then
    return 0
  fi
  load_env
  verify_toolchain
}

# ── Toolchain verification ──────────────────────────────────────────
verify_toolchain() {
  if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found. Install Rust via https://rustup.rs" >&2
    exit 1
  fi

  # Check that the toolchain specified in rust-toolchain.toml is installed
  if [ -f "$REPO_ROOT/rust-toolchain.toml" ]; then
    local channel
    channel=$(grep -oP 'channel\s*=\s*"\K[^"]+' "$REPO_ROOT/rust-toolchain.toml" || echo "")
    if [ -n "$channel" ]; then
      if ! rustup toolchain list | grep -q "$channel"; then
        echo "==> Installing toolchain $channel..."
        rustup toolchain install "$channel"
      fi
    fi
  fi

  echo "==> Toolchain: $(rustc --version)"
}

# ── Load .env ────────────────────────────────────────────────────────
load_env() {
  if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
  fi
}

# ── Workspace crate names ───────────────────────────────────────────
# Maps suite/short names to cargo package names
crate_package() {
  case "$1" in
    core)      echo "rustml-core" ;;
    nn)        echo "rustml-nn" ;;
    nlp)       echo "rustml-nlp" ;;
    tokenizer) echo "rustml-tokenizer" ;;
    gguf)      echo "rustml-gguf" ;;
    quant)     echo "rustml-quant" ;;
    hub)       echo "rustml-hub" ;;
    *)         echo "$1" ;;
  esac
}
