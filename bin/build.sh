#!/usr/bin/env bash
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"

BUILD_MODE=""
CRATE=""
for arg in "$@"; do
  case "$arg" in
    --release) BUILD_MODE="--release" ;;
    --debug)   BUILD_MODE="" ;;
    -p)        CRATE="__next__" ;;
    *)
      if [ "$CRATE" = "__next__" ]; then
        CRATE="$arg"
      fi
      ;;
  esac
done

PROFILE_LABEL="${BUILD_MODE:+release}"
PROFILE_LABEL="${PROFILE_LABEL:-debug}"

preflight

if [ -n "$CRATE" ] && [ "$CRATE" != "__next__" ]; then
  PKG=$(crate_package "$CRATE")
  echo "==> Building $PKG ($PROFILE_LABEL)..."
  cargo build -p "$PKG" $BUILD_MODE
else
  echo "==> Building workspace ($PROFILE_LABEL)..."
  cargo build --workspace $BUILD_MODE
fi

echo "==> Build complete ($PROFILE_LABEL)"
