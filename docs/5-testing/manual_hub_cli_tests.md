# Manual Hub CLI Tests

> **TLDR:** Manual test checklist for the `rustml-hub-cli`: download models, list cache, and inspect configs.

**Audience**: Developers, QA

**WHAT**: Manual test procedures for the HuggingFace Hub CLI tool
**WHY**: Validates model downloading, caching, and config inspection for the local model management workflow
**HOW**: Step-by-step test tables with expected outcomes

---

## Table of Contents

- [Help & Version](#1-help--version)
- [Global Flags](#2-global-flags)
- [Download — SafeTensors](#3-download--safetensors)
- [Download — GGUF](#4-download--gguf)
- [List](#5-list)
- [Info](#6-info)
- [Error Cases](#7-error-cases)

---

> **Prerequisite**: Internet access for download tests. A HuggingFace API token for private model tests (optional).
>
> **Unified CLI**: All `rustml-hub-cli` commands below can also be run as `sweai hub` (e.g., `sweai hub download openai-community/gpt2`). Build with `cargo build -p rustml-cli`.

## 1. Help & Version

| Test | Command | Expected |
|------|---------|----------|
| Help flag | `rustml-hub-cli --help` | Lists `download`, `list`, `info` subcommands and `--cache-dir`, `--token` global flags |
| Version flag | `rustml-hub-cli --version` | Prints version string |
| No args | `rustml-hub-cli` | Shows error and usage |
| Download help | `rustml-hub-cli download --help` | Shows `MODEL_ID` positional and `--gguf` option |
| List help | `rustml-hub-cli list --help` | Shows list usage |
| Info help | `rustml-hub-cli info --help` | Shows `MODEL_ID` positional |
| Unified help | `sweai hub --help` | Same subcommands and flags as `rustml-hub-cli --help` |

## 2. Global Flags

| Test | Command | Expected |
|------|---------|----------|
| Custom cache dir | `rustml-hub-cli --cache-dir /tmp/test-cache list` | Uses specified cache directory |
| Token flag | `rustml-hub-cli --token hf_xxx list` | Accepts token (used for private repos) |
| Flags before subcommand | `rustml-hub-cli --cache-dir /tmp/c --token hf_xxx list` | Both flags parsed correctly |

## 3. Download — SafeTensors

> **Note**: These tests download files from HuggingFace. Use a small model like `openai-community/gpt2` (~500 MB).

| Test | Command | Expected |
|------|---------|----------|
| Download model | `rustml-hub-cli download openai-community/gpt2` | Downloads config.json, model.safetensors, vocab files; prints model directory path |
| Cached rerun | Run same command again | Completes quickly (files already cached); prints same path |
| Custom cache | `rustml-hub-cli --cache-dir /tmp/hub-test download openai-community/gpt2` | Downloads to specified directory |

## 4. Download — GGUF

> **Note**: GGUF downloads require a repo with GGUF files and can be large.

| Test | Command | Expected |
|------|---------|----------|
| Download GGUF | `rustml-hub-cli download <repo-id> --gguf <filename>.gguf` | Downloads single GGUF file; prints file path |
| GGUF in list after download | `rustml-hub-cli list` (after GGUF download) | Shows the GGUF repo ID in the cached models list |

## 5. List

| Test | Command | Expected |
|------|---------|----------|
| List after download | `rustml-hub-cli list` | Shows model IDs reconstructed from cache directories |
| List with custom cache | `rustml-hub-cli --cache-dir /tmp/hub-test list` | Shows models in custom cache |
| List empty cache | `rustml-hub-cli --cache-dir /tmp/empty-dir list` | `No cached models found.` on stderr |
| List nonexistent dir | `rustml-hub-cli --cache-dir /tmp/nonexistent list` | `Cache directory does not exist` on stderr |
| Model ID format | (in list output) | IDs shown as `org/model` (not `org--model`) |
| GGUF-only entries | (in list output) | GGUF-only cache entries (no config.json) also appear in list |

## 6. Info

| Test | Command | Expected |
|------|---------|----------|
| Show config | `rustml-hub-cli info openai-community/gpt2` | Pretty-printed JSON with `model_type`, `hidden_size`, etc. |
| Valid JSON | (same output) | Output parses as valid JSON |
| Custom cache | `rustml-hub-cli --cache-dir /tmp/hub-test info openai-community/gpt2` | Shows config from custom cache |

## 7. Error Cases

| Test | Command | Expected |
|------|---------|----------|
| Info uncached model | `rustml-hub-cli info nonexistent/model` | Error: model not cached |
| Download bad model | `rustml-hub-cli download nonexistent/model-xyz-99999` | Error: download failed |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites and setup
- [Manual Inference Tests](manual_infer_tests.md) — loading models for inference
- [Manual GGUF Inspector Tests](manual_gguf_inspect_tests.md) — inspecting downloaded GGUF files
