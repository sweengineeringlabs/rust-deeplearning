# Manual Testing Guide

> **TLDR:** Hub for all manual testing procedures — links to focused test documents by domain.

**Audience**: Developers, QA

**WHAT**: Central navigation hub for manual test procedures
**WHY**: Provides a single entry point so testers can find the right checklist for their task
**HOW**: Organized by domain with shared prerequisites and setup instructions

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Running the CLI](#running-the-cli)
- [Test Documents](#test-documents)
- [Automated Test Suites](#automated-test-suites)

---

## Prerequisites

1. Rust toolchain (stable recommended; the parent repo pins 1.85 but its `cargo` may be missing)
2. Internet connection for HuggingFace model downloads (AI tests)
3. ~2 GB disk space for model files (GPT-2 124M + TinyLlama 1.1B GGUF)

## Running the CLI

```bash
# Via the launcher script
./llmf build
./llmf run -- run --help

# Via cargo directly
RUSTUP_TOOLCHAIN=stable cargo run -p llmforge-cli -- --help

# Pre-download models for offline testing
./llmf run -- download openai-community/gpt2
./llmf run -- download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --files tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```

### Scripted (Non-Interactive) Testing

Most manual tests can be scripted by piping prompts or using `--prompt` for single-shot generation.

```bash
# Single-shot generation (no interactive terminal needed)
./llmf run -- run --model openai-community/gpt2 \
  --prompt "Hello world" --max-tokens 16

# Info command (fully non-interactive)
./llmf run -- info --model openai-community/gpt2

# Download (fully non-interactive)
./llmf run -- download openai-community/gpt2
```

**Notes:**
- Status messages print to stderr, generated text to stdout. Use `2>/dev/null` to isolate generated output.
- First run for a given model triggers a HuggingFace download (may take minutes depending on connection speed).
- Subsequent runs use cached files from `~/.cache/huggingface/`.

---

## Test Documents

| Document | Domain | Tests |
|----------|--------|-------|
| [Launcher Tests](manual_launcher_tests.md) | llmf help, build, test, run dispatch | 19 |
| [CLI Tests](manual_cli_tests.md) | CLI commands: run, info, download, argument handling | 48 |
| [Model Loading Tests](manual_model_tests.md) | Model source resolution, format detection, arch auto-detect | 37 |
| [Inference Tests](manual_inference_tests.md) | Text generation, sampling parameters, REPL mode | 34 |

---

## Automated Test Suites

For reference, the automated tests cover these areas:

```bash
# Library unit + integration tests (no model download needed)
./llmf test lib

# CLI compile check + help verification
./llmf test cli

# All tests
./llmf test
```

| Suite | Location | Count |
|-------|----------|-------|
| Sampling unit tests | `src/inference/sampling.rs` | 9 |
| Arena tests | `tests/arena_test.rs` | 5 |
| Attention tests | `tests/attention_test.rs` | 7 |
| Cache layout tests | `tests/cache_layout_test.rs` | 4 |
| DType conversion tests | `tests/dtype_conversion_test.rs` | 4 |
| Generation tests | `tests/generation_test.rs` | 9 |
| Inference tests | `tests/inference_test.rs` | 1 |
| Integration tests | `tests/integration_test.rs` | 7 |
| Loader tests | `tests/loader_test.rs` | 10 |
| Model architecture tests | `tests/model_arch_test.rs` | 33 |
| Model compatibility tests | `tests/model_compat_test.rs` | 33 |
| Model tests | `tests/model_test.rs` | 1 |
| NN layer tests | `tests/nn_layers_test.rs` | 10 |
| Parallelism tests | `tests/parallelism_test.rs` | 2 |
| Quantization Q4 tests | `tests/quantization_q4_test.rs` | 10 |
| Quantization tests | `tests/quantization_test.rs` | 7 |
| SafeTensors tests | `tests/safetensors_test.rs` | 1 |
| Safety tests | `tests/safety_test.rs` | 21 |
| SIMD tests | `tests/simd_test.rs` | 8 |
| Tensor ops extended tests | `tests/tensor_ops_extended_test.rs` | 15 |
| Tensor ops tests | `tests/tensor_ops_test.rs` | 11 |
| Tensor views tests | `tests/tensor_views_test.rs` | 9 |
| Transformer tests | `tests/transformer_test.rs` | 1 |
| Weight loading tests | `tests/weight_loading_test.rs` | 7 |
| **Total** | | **225** |
