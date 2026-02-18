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
4. (Optional) Internet access for Hub download tests

## Building the CLIs

```bash
# Build all four CLI binaries (debug)
cargo build -p rustml-tokenizer -p rustml-gguf -p rustml-nlp -p rustml-hub

# Build all four CLI binaries (release, recommended for inference)
cargo build --release -p rustml-tokenizer -p rustml-gguf -p rustml-nlp -p rustml-hub

# Verify binaries exist
ls target/debug/rustml-tokenizer target/debug/rustml-gguf-inspect target/debug/rustml-hub-cli target/debug/rustml-infer
```

The four CLI binaries:

| Binary | Package | Purpose |
|--------|---------|---------|
| `rustml-tokenizer` | `rustml-tokenizer` | Encode, decode, and inspect tokenizer vocabularies |
| `rustml-gguf-inspect` | `rustml-gguf` | Inspect GGUF model files (metadata, tensors, config) |
| `rustml-hub-cli` | `rustml-hub` | Download and manage HuggingFace models |
| `rustml-infer` | `rustml-nlp` | Run text generation on GGUF models |

---

## Test Documents

| Document | Domain | Tests |
|----------|--------|-------|
| [Tokenizer Tests](manual_tokenizer_tests.md) | Tokenizer encode/decode, backends, vocab info | 28 |
| [GGUF Inspector Tests](manual_gguf_inspect_tests.md) | GGUF metadata, tensor listing, model info | 25 |
| [Hub CLI Tests](manual_hub_cli_tests.md) | Model download, cache listing, config display | 18 |
| [Inference Tests](manual_infer_tests.md) | GGUF model loading, text generation, streaming, batch, timeout | 52 |

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
| GGUF unit tests | `rustml/gguf/main/src/` | 5 |
| GGUF CLI integration | `rustml/gguf/tests/cli.rs` | 27 |
| Hub unit tests | `rustml/hub/main/src/` | 8 |
| Hub CLI integration | `rustml/hub/tests/cli.rs` | 15 |
| NLP unit tests | `rustml/nlp/main/src/` | 57 |
| NLP integration tests | `rustml/nlp/tests/` | 4 |
| Infer CLI integration | `rustml/nlp/tests/infer_cli.rs` | 13 |
| Tokenizer CLI integration | `rustml/tokenizer/tests/cli.rs` | 18 |
| **Total** | | **147** |

---

## See Also

- [Architecture](../3-design/architecture.md) — project structure and crate layout
- [Model Verification Guide](../4-development/guides/model-verification.md) — verifying model correctness
