#!/usr/bin/env bash
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"

preflight

BINARY=""
CARGO_BUILD_FLAG=""
PROFILE_DIR="debug"
PASSTHROUGH_ARGS=()
PAST_SEPARATOR=false

for arg in "$@"; do
  if [ "$PAST_SEPARATOR" = true ]; then
    PASSTHROUGH_ARGS+=("$arg")
    continue
  fi
  case "$arg" in
    --release) PROFILE_DIR="release"; CARGO_BUILD_FLAG="--release" ;;
    --debug)   PROFILE_DIR="debug";   CARGO_BUILD_FLAG="" ;;
    --)        PAST_SEPARATOR=true ;;
    *)
      if [ -z "$BINARY" ]; then
        BINARY="$arg"
      else
        PASSTHROUGH_ARGS+=("$arg")
      fi
      ;;
  esac
done

if [ -z "$BINARY" ]; then
  echo "Usage: ./rdl run <BINARY> [--release] [-- args...]" >&2
  echo "" >&2
  echo "Available binaries:" >&2
  echo "  rustml-tokenizer    Tokenizer CLI (encode/decode/info)" >&2
  exit 1
fi

# Build if the binary doesn't exist yet
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BIN_PATH="$TARGET_DIR/$PROFILE_DIR/$BINARY"

if [ ! -f "$BIN_PATH" ]; then
  echo "==> Binary not found, building ($PROFILE_DIR)..." >&2
  cargo build --bin "$BINARY" $CARGO_BUILD_FLAG
fi

echo "==> Running $BINARY ($PROFILE_DIR)..." >&2
exec "$BIN_PATH" "${PASSTHROUGH_ARGS[@]}"
