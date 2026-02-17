#!/usr/bin/env bash
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"

preflight

PROFILE_DIR="debug"
CARGO_BUILD_FLAG=""
CLI_ARGS=()
PAST_SEPARATOR=false

for arg in "$@"; do
  if [ "$PAST_SEPARATOR" = true ]; then
    CLI_ARGS+=("$arg")
  elif [ "$arg" = "--" ]; then
    PAST_SEPARATOR=true
  else
    case "$arg" in
      --release) PROFILE_DIR="release"; CARGO_BUILD_FLAG="--release" ;;
      *)         CLI_ARGS+=("$arg") ;;
    esac
  fi
done

CLI_BIN="$TARGET_DIR/$PROFILE_DIR/llmforge"

if [ ! -f "$CLI_BIN" ]; then
  echo "==> Binary not found, building ($PROFILE_DIR)..."
  cargo build --manifest-path "$REPO_ROOT/Cargo.toml" -p llmforge-cli $CARGO_BUILD_FLAG
fi

echo "==> Launching llmforge ($PROFILE_DIR)..."
exec "$CLI_BIN" "${CLI_ARGS[@]}"
