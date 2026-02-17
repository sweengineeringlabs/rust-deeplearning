#!/usr/bin/env bash
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"

PLATFORM=$(detect_platform)
echo "==> Detected platform: $PLATFORM"

# ── Check prerequisites ─────────────────────────────────────────────
echo "==> Checking prerequisites..."

if ! command -v rustup &>/dev/null; then
  echo "ERROR: rustup not found. Install from https://rustup.rs" >&2
  exit 1
fi

if ! command -v cargo &>/dev/null; then
  echo "ERROR: cargo not found (toolchain=$LLMFORGE_TOOLCHAIN)." >&2
  echo "  Try: rustup toolchain install $LLMFORGE_TOOLCHAIN" >&2
  exit 1
fi

echo "  rustup:    $(rustup --version 2>&1 | head -1)"
echo "  cargo:     $(cargo --version)"
echo "  toolchain: $LLMFORGE_TOOLCHAIN"

# ── Verify workspace builds ─────────────────────────────────────────
echo "==> Verifying workspace build..."
cargo build -p llmforge-cli --manifest-path "$REPO_ROOT/Cargo.toml"

echo ""
echo "Setup complete!"
echo "  Platform:  $PLATFORM"
echo "  Toolchain: $LLMFORGE_TOOLCHAIN"
echo ""
echo "Next steps:"
echo "  ./llmf build              # build (debug)"
echo "  ./llmf build --release    # build (release)"
echo "  ./llmf run -- run --help  # see run subcommand help"
