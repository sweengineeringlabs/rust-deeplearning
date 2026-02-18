#!/usr/bin/env bash
# Unit tests for lib/common.sh

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TESTS_DIR/../.." && pwd)"

# -- detect_platform --------------------------------------------------

test_detect_platform_returns_valid_platform() {
  local result
  result=$(
    set +euo pipefail
    source "$REPO_ROOT/lib/common.sh"
    detect_platform
  )
  assert_match "$result" "^(wsl|linux|macos|mingw)$" "detect_platform should return wsl, linux, macos, or mingw"
}

# -- verify_toolchain -------------------------------------------------

test_verify_toolchain_succeeds_when_cargo_exists() {
  local tmpout tmperr
  tmpout=$(mktemp); tmperr=$(mktemp)
  (
    set +euo pipefail
    source "$REPO_ROOT/lib/common.sh"
    verify_toolchain
  ) >"$tmpout" 2>"$tmperr"
  local ec=$?
  local out=$(cat "$tmpout")
  rm -f "$tmpout" "$tmperr"
  assert_exit_code 0 "$ec" "verify_toolchain should succeed when cargo is installed"
  assert_contains "$out" "Toolchain:" "should print toolchain version"
}

# -- load_env ----------------------------------------------------------

test_load_env_sets_vars_from_env_file() {
  local tmpdir result
  tmpdir=$(mktemp -d)
  cp -r "$REPO_ROOT/lib" "$tmpdir/lib"
  echo 'TEST_VAR_FROM_ENV=hello_from_env' > "$tmpdir/.env"

  result=$(
    set +euo pipefail
    REPO_ROOT="$tmpdir"
    source "$tmpdir/lib/common.sh"
    REPO_ROOT="$tmpdir"
    load_env
    echo "$TEST_VAR_FROM_ENV"
  )
  rm -rf "$tmpdir"
  assert_eq "hello_from_env" "$result" "load_env should set vars from .env"
}

test_load_env_ignores_comment_lines() {
  local tmpdir result
  tmpdir=$(mktemp -d)
  cp -r "$REPO_ROOT/lib" "$tmpdir/lib"
  printf '# this is a comment\nACTUAL_VAR=real_value\n' > "$tmpdir/.env"

  result=$(
    set +euo pipefail
    REPO_ROOT="$tmpdir"
    source "$tmpdir/lib/common.sh"
    REPO_ROOT="$tmpdir"
    load_env
    echo "$ACTUAL_VAR"
  )
  rm -rf "$tmpdir"
  assert_eq "real_value" "$result" "load_env should ignore comments and set real vars"
}

test_load_env_handles_quoted_values() {
  local tmpdir result
  tmpdir=$(mktemp -d)
  cp -r "$REPO_ROOT/lib" "$tmpdir/lib"
  echo 'QUOTED_VAR="hello world"' > "$tmpdir/.env"

  result=$(
    set +euo pipefail
    REPO_ROOT="$tmpdir"
    source "$tmpdir/lib/common.sh"
    REPO_ROOT="$tmpdir"
    load_env
    echo "$QUOTED_VAR"
  )
  rm -rf "$tmpdir"
  assert_eq "hello world" "$result" "load_env should handle quoted values"
}

test_load_env_noops_when_file_missing() {
  local tmpdir
  tmpdir=$(mktemp -d)
  cp -r "$REPO_ROOT/lib" "$tmpdir/lib"

  local ec
  (
    set +euo pipefail
    REPO_ROOT="$tmpdir"
    source "$tmpdir/lib/common.sh"
    REPO_ROOT="$tmpdir"
    load_env
  )
  ec=$?
  rm -rf "$tmpdir"
  assert_exit_code 0 "$ec" "load_env should no-op when .env is missing"
}

# -- crate_package -----------------------------------------------------

test_crate_package_maps_short_names() {
  local result
  result=$(
    set +euo pipefail
    source "$REPO_ROOT/lib/common.sh"
    crate_package "core"
  )
  assert_eq "rustml-core" "$result" "crate_package core should map to rustml-core"
}

test_crate_package_maps_tokenizer() {
  local result
  result=$(
    set +euo pipefail
    source "$REPO_ROOT/lib/common.sh"
    crate_package "tokenizer"
  )
  assert_eq "rustml-tokenizer" "$result" "crate_package tokenizer should map to rustml-tokenizer"
}

test_crate_package_passes_through_unknown() {
  local result
  result=$(
    set +euo pipefail
    source "$REPO_ROOT/lib/common.sh"
    crate_package "some-other-crate"
  )
  assert_eq "some-other-crate" "$result" "crate_package should pass through unknown names"
}
