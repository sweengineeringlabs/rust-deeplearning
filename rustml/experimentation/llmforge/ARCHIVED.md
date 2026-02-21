# ARCHIVED — llmforge Prototype

> **Status: Archived (read-only)**
>
> This codebase is preserved for historical reference only. No further
> development or maintenance will occur here. See the production equivalents
> below.

---

## What this was

`llmforge` was a self-contained experimental prototype for LLM inference in Rust.
It explored tensor ops, GGUF parsing, HuggingFace Hub integration, tokenization,
and a CLI (`llmforge` binary + `./llmf` launcher) from scratch.

It was never migrated to depend on the `rustml/` workspace crates — the two
implementations are independent.

## Production equivalents

| llmforge module | Production crate | Binary |
|---|---|---|
| `src/core/tensor/` | `rustml-core` | — |
| `src/attention/`, `src/transformer/` | `rustml-nn` | — |
| `src/gguf/` | `rustml-gguf` | `rustml-gguf-inspect` |
| `src/hub/` | `rustml-hub` | `rustml-hub-cli` |
| `src/tokenizer/` | `rustml-tokenizer` | `rustml-tokenizer` |
| `src/inference/`, `src/model/` | `rustml-nlp` | `rustml-infer` |
| `cli/` | `rustml-cli` | `sweai` |

## Manual test document migration

The llmforge manual test docs in `docs/testing/` have been superseded by the
production test suite in `rust-deeplearning/docs/5-testing/`. The mapping is:

| llmforge doc | Tests | Production equivalent | Notes |
|---|---|---|---|
| `manual_launcher_tests.md` | 19 | `manual_testing.md` (Building the CLIs) | `./llmf` commands map to `cargo build -p rustml-cli` and `sweai` |
| `manual_cli_tests.md` | 48 | `manual_sweai_tests.md` (77 tests) + `manual_infer_tests.md` | Production docs are more comprehensive |
| `manual_model_tests.md` | 37 | `manual_infer_tests.md` §10–12 (SafeTensors), §2 (GGUF local) | Directory-based model resolution was llmforge-specific; not in production CLI |
| `manual_inference_tests.md` | 34 | `manual_infer_tests.md` (143 tests) | Production docs cover all inference scenarios plus batch, interactive, profiling, streaming |

**Coverage delta:** The one llmforge-specific test area with no direct production
equivalent is directory-based model resolution (pointing `--model ./dir/` at a
directory and having the CLI auto-select the `.gguf` or `model.safetensors`
inside it). The production `rustml-infer` uses an explicit `[GGUF_PATH]`
positional argument instead.

## Decision record

See `docs/3-design/adr/adr-002-retire-llmforge-prototype.md` for the full
decision rationale.
