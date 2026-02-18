#!/usr/bin/env bash
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"

PLATFORM=$(detect_platform)
echo "==> Detected platform: $PLATFORM"

# ── Check prerequisites ──────────────────────────────────────────────
echo "==> Checking prerequisites..."

if ! command -v rustup &>/dev/null; then
  echo "ERROR: rustup not found. Install from https://rustup.rs" >&2
  exit 1
fi

if ! command -v cargo &>/dev/null; then
  echo "ERROR: cargo not found. Install Rust via rustup." >&2
  exit 1
fi

echo "  rustup: $(rustup --version 2>&1 | head -1)"
echo "  cargo:  $(cargo --version)"
echo "  rustc:  $(rustc --version)"

# ── Install required toolchain ──────────────────────────────────────
if [ -f "$REPO_ROOT/rust-toolchain.toml" ]; then
  CHANNEL=$(grep -oP 'channel\s*=\s*"\K[^"]+' "$REPO_ROOT/rust-toolchain.toml" || echo "")
  if [ -n "$CHANNEL" ]; then
    echo "==> Installing toolchain $CHANNEL..."
    rustup toolchain install "$CHANNEL"

    # Install components listed in rust-toolchain.toml
    COMPONENTS=$(grep -oP 'components\s*=\s*\[\K[^\]]+' "$REPO_ROOT/rust-toolchain.toml" | tr -d '"' | tr ',' '\n' | tr -d ' ')
    for comp in $COMPONENTS; do
      echo "  Installing component: $comp"
      rustup component add "$comp" --toolchain "$CHANNEL" 2>/dev/null || true
    done
  fi
fi

# ── Verify workspace builds ─────────────────────────────────────────
echo "==> Verifying workspace builds..."
cargo check --workspace 2>&1 | tail -1

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "Setup complete!"
echo "  Platform:   $PLATFORM"
echo "  Toolchain:  $(rustc --version)"
echo ""
echo "Next steps:"
echo "  ./rdl build           Build the workspace"
echo "  ./rdl test            Run all tests"
echo "  ./rdl test tokenizer  Run tokenizer tests"
