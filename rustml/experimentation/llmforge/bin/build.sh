#!/usr/bin/env bash
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"

BUILD_MODE=""
for arg in "$@"; do
  case "$arg" in
    --release) BUILD_MODE="--release" ;;
  esac
done

PROFILE_LABEL="${BUILD_MODE:+release}"
PROFILE_LABEL="${PROFILE_LABEL:-debug}"

preflight

echo "==> Building llmforge library..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml" -p llmforge $BUILD_MODE

echo "==> Building llmforge-cli ($PROFILE_LABEL)..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml" -p llmforge-cli $BUILD_MODE

echo "==> Build complete ($PROFILE_LABEL)"
