# Manual Launcher Tests

> **TLDR:** Manual test checklist for the llmf launcher: help, build, test dispatch, and run.

**Audience**: Developers, QA

**WHAT**: Manual test procedures for the llmf launcher and its subcommands
**WHY**: Validates the primary entry point delegates correctly to build, test, and run scripts
**HOW**: Step-by-step test tables with expected outcomes

---

## Table of Contents

- [llmf help](#1-llmf-help)
- [llmf setup](#2-llmf-setup)
- [llmf build](#3-llmf-build)
- [llmf test](#4-llmf-test)
- [llmf run](#5-llmf-run)
- [llmf shorthands](#6-llmf-shorthands)

---

## 1. llmf help

| Test | Command | Expected |
|------|---------|----------|
| Help flag | `./llmf --help` | Prints usage with all commands: setup, build, run, test, info, download |
| Help command | `./llmf help` | Same output as `--help` |
| No args | `./llmf` | Prints usage and exits with code 0 |
| Unknown command | `./llmf foo` | Prints usage and exits with code 1 |

## 2. llmf setup

| Test | Command | Expected |
|------|---------|----------|
| Prerequisites check | `./llmf setup` | Prints platform, rustup version, cargo version, toolchain |
| Build verification | `./llmf setup` | Runs `cargo build -p llmforge-cli` successfully |
| Missing rustup | Rename rustup temporarily, run `./llmf setup` | Prints `ERROR: rustup not found`, exits 1 |

## 3. llmf build

| Test | Command | Expected |
|------|---------|----------|
| Debug build (default) | `./llmf build` | Builds library and CLI in debug mode, prints `Build complete (debug)` |
| Release build | `./llmf build --release` | Builds library and CLI in release mode, prints `Build complete (release)` |
| Idempotent | Run `./llmf build` twice | Second build completes quickly (incremental), no errors |

## 4. llmf test

| Test | Command | Expected |
|------|---------|----------|
| All suites | `./llmf test` | Runs lib tests then CLI compile check; all pass |
| Lib only | `./llmf test lib` | Runs `cargo test -p llmforge` only; 225 tests pass |
| CLI only | `./llmf test cli` | Builds CLI, verifies `--help` for all subcommands |
| Unknown suite | `./llmf test foo` | Prints usage and exits with code 1 |

## 5. llmf run

| Test | Command | Expected |
|------|---------|----------|
| Auto-build | Delete `target/debug/llmforge`, run `./llmf run -- --help` | Builds CLI first, then shows help |
| Pass-through args | `./llmf run -- run --help` | Shows `run` subcommand help with all flags |
| Release mode | `./llmf run --release -- --help` | Uses release binary, shows help |

## 6. llmf shorthands

| Test | Command | Expected |
|------|---------|----------|
| Info shorthand | `./llmf info --model ./model.gguf` | Equivalent to `./llmf run -- info --model ./model.gguf` |
| Download shorthand | `./llmf download openai-community/gpt2` | Equivalent to `./llmf run -- download openai-community/gpt2` |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites and setup
- [Manual CLI Tests](manual_cli_tests.md) — CLI command tests
