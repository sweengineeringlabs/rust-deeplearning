#!/usr/bin/env bash
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"

CHECK_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_FLAG="--check" ;;
  esac
done

echo "==> Running cargo fmt${CHECK_FLAG:+ (check mode)}..."
cargo fmt --all $CHECK_FLAG
echo "==> Done"
