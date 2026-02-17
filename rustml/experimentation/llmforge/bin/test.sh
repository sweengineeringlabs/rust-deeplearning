#!/usr/bin/env bash
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"

SUITE="${1:-all}"

preflight

run_lib_tests() {
  echo "==> Testing llmforge library..."
  cargo test --manifest-path "$REPO_ROOT/Cargo.toml" -p llmforge
}

run_cli_tests() {
  echo "==> Building llmforge-cli (compile check)..."
  cargo build --manifest-path "$REPO_ROOT/Cargo.toml" -p llmforge-cli

  echo "==> Verifying CLI help output..."
  "$TARGET_DIR/debug/llmforge" --help >/dev/null
  "$TARGET_DIR/debug/llmforge" run --help >/dev/null
  "$TARGET_DIR/debug/llmforge" info --help >/dev/null
  "$TARGET_DIR/debug/llmforge" download --help >/dev/null
  echo "  All help commands OK"
}

case "$SUITE" in
  lib)  run_lib_tests ;;
  cli)  run_cli_tests ;;
  all)
    run_lib_tests
    run_cli_tests
    ;;
  *)
    echo "Usage: ./llmf test [lib|cli|all]"
    exit 1
    ;;
esac

echo "==> Tests complete ($SUITE)"
