# Manual Testing Guide

> **TLDR:** Hub for all manual testing procedures — links to focused test documents by CLI tool.

**Audience**: Developers, QA

**WHAT**: Central navigation hub for manual test procedures across all RustML CLI tools
**WHY**: Provides a single entry point so testers can find the right checklist for their CLI
**HOW**: Organized by CLI binary with shared prerequisites and setup instructions

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building the CLIs](#building-the-clis)
- [Test Documents](#test-documents)
- [Automated Test Suites](#automated-test-suites)

---

## Prerequisites

1. Rust toolchain (edition 2024, stable channel 1.93+)
2. The workspace builds cleanly: `cargo build --workspace`
3. (Optional) A GGUF model file for inference and inspection tests (e.g., Gemma 3 1B Q4_0)
4. (Optional) A cached SafeTensors model for SafeTensors inference tests (e.g., `sweai hub download openai-community/gpt2`)
5. (Optional) Internet access for Hub download tests
6. (Optional) A HuggingFace API token for gated model tests (e.g., `google/gemma-3-1b-it`)

### HuggingFace Token Setup

A token is required to download gated models (e.g., Gemma 3). Token precedence (highest to lowest):

| Priority | Source | Example |
|----------|--------|---------|
| 1 | `--token` CLI flag | `sweai hub --token hf_xxx download google/gemma-3-1b-it` |
| 2 | `HF_TOKEN` environment variable | `export HF_TOKEN=hf_xxx` |
| 3 | Token file | `~/.cache/huggingface/token` (written by `huggingface-cli login`) |

For inference with gated models, set the environment variable before running:

```bash
export HF_TOKEN=hf_xxx
sweai infer --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 20
```

> **Note:** The `--token` flag is available on `sweai hub` and `rustml-hub-cli` but not on `sweai infer` / `rustml-infer`. For inference, use `HF_TOKEN` env var or `~/.cache/huggingface/token`.

## Building the CLIs

```bash
# Build all four standalone CLI binaries (debug)
cargo build -p rustml-tokenizer -p rustml-gguf -p rustml-nlp -p rustml-hub

# Build the unified sweai binary (debug)
cargo build -p rustml-cli

# Build everything (release, recommended for inference)
cargo build --release -p rustml-tokenizer -p rustml-gguf -p rustml-nlp -p rustml-hub -p rustml-cli

# Verify binaries exist
ls target/debug/rustml-tokenizer target/debug/rustml-gguf-inspect target/debug/rustml-hub-cli target/debug/rustml-infer target/debug/sweai
```

### Unified CLI: `sweai`

The `sweai` binary is a single facade over all four standalone CLIs. Every standalone command has a `sweai` equivalent:

| Standalone | Unified equivalent |
|------------|-------------------|
| `rustml-infer ...` | `sweai infer ...` |
| `rustml-gguf-inspect ...` | `sweai gguf ...` |
| `rustml-hub-cli ...` | `sweai hub ...` |
| `rustml-tokenizer ...` | `sweai tokenizer ...` |

### Standalone binaries

| Binary | Package | Purpose |
|--------|---------|---------|
| `rustml-tokenizer` | `rustml-tokenizer` | Encode, decode, and inspect tokenizer vocabularies |
| `rustml-gguf-inspect` | `rustml-gguf` | Inspect GGUF model files (metadata, tensors, config) |
| `rustml-hub-cli` | `rustml-hub` | Download and manage HuggingFace models |
| `rustml-infer` | `rustml-nlp` | Run text generation on GGUF or SafeTensors models |
| `sweai` | `rustml-cli` | Unified CLI with `infer`, `gguf`, `hub`, `tokenizer` subcommands |

---

## Test Documents

| Document | Domain | Tests |
|----------|--------|-------|
| [SweAI Unified CLI Tests](manual_sweai_tests.md) | Top-level help, subcommand dispatch, parity, error handling | 70 |
| [Tokenizer Tests](manual_tokenizer_tests.md) | Tokenizer encode/decode, backends, vocab info | 28 |
| [GGUF Inspector Tests](manual_gguf_inspect_tests.md) | GGUF metadata, tensor listing, model info | 25 |
| [Hub CLI Tests](manual_hub_cli_tests.md) | Model download, cache listing, config display, SafeTensors inference | 30 |
| [Inference Tests](manual_infer_tests.md) | GGUF and SafeTensors model loading, multi-arch dispatch, text generation, streaming, batch, timeout, optimization profiles | 92 |

> **Note:** Each test document lists standalone binary commands. All commands can also be run via the unified `sweai` binary — see the equivalence table above.

---

## Automated Test Suites

For reference, the automated tests cover these areas:

```bash
# Unit + integration tests (no model files needed)
cargo test --workspace

# Individual crate tests
cargo test -p rustml-gguf
cargo test -p rustml-hub
cargo test -p rustml-nlp
cargo test -p rustml-tokenizer
```

| Suite | Location | Count |
|-------|----------|-------|
| Core unit tests | `rustml/core/main/src/` | 44 |
| NN unit tests | `rustml/nn/main/src/` | 69 |
| GGUF unit tests | `rustml/gguf/main/src/` | 5 |
| GGUF CLI integration | `rustml/gguf/tests/cli.rs` | 27 |
| Hub unit tests | `rustml/hub/main/src/` | 8 |
| Hub CLI integration | `rustml/hub/tests/cli.rs` | 15 |
| NLP unit tests | `rustml/nlp/main/src/` | 69 |
| NLP integration tests | `rustml/nlp/tests/` | 4 |
| Infer CLI integration | `rustml/nlp/tests/infer_cli.rs` | 14 |
| Tokenizer CLI integration | `rustml/tokenizer/tests/cli.rs` | 18 |
| **Total** | | **273** |

---

## See Also

- [Architecture](../3-design/architecture.md) — project structure and crate layout
- [Model Verification Guide](../4-development/guides/model-verification.md) — verifying model correctness
