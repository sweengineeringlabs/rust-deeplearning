# ADR-002: Retire llmforge Prototype — Superseded by rustml Workspace

**Status:** Accepted
**Date:** 2026-02-21

## Context

`experimentation/llmforge/` is a self-contained experimental prototype that was
developed to explore LLM inference in Rust. It implements its own tensor engine,
GGUF parser, HuggingFace Hub client, tokenizer, and CLI from scratch, using
third-party crates directly (`faer`, `safetensors`, `tokenizers`, `hf-hub`).

In parallel, the `rustml/` workspace grew into a production-quality rewrite of the
same domain, structured as a multi-crate workspace following the Stratified
Encapsulation Architecture (SEA) pattern:

| llmforge module | Production equivalent |
|---|---|
| `src/core/tensor/` | `rustml-core` (`rustml/core/`) |
| `src/attention/`, `src/transformer/` | `rustml-nn` (`rustml/nn/`) |
| `src/gguf/` | `rustml-gguf` (`rustml/gguf/`) |
| `src/hub/` | `rustml-hub` (`rustml/hub/`) |
| `src/tokenizer/` | `rustml-tokenizer` (`rustml/tokenizer/`) |
| `src/inference/`, `src/model/` | `rustml-nlp` (`rustml/nlp/`) |
| `cli/` (`llmforge` binary) | `rustml-cli` (`sweai` binary) |

The two implementations never converged — llmforge was never migrated to depend
on the rustml workspace crates. They are independent codebases sharing no Rust
source.

Key differences:

| Aspect | llmforge | rustml workspace |
|---|---|---|
| Workspace | Separate (`edition = "2021"`) | Part of monorepo workspace (`edition = "2024"`) |
| Architecture | Flat `src/` modules | SEA multi-crate (api/core/spi layers) |
| Launcher | `./llmf` bash script | `sweai` unified CLI + standalone binaries |
| GGUF support | Implemented in-crate | `rustml-gguf` crate |
| SafeTensors multi-arch | GPT-2, Llama | GPT-2, Llama, Mistral, Qwen2, Phi3, Gemma3, Falcon, Mixtral |
| Interactive mode | REPL (basic) | Multi-turn chat with `/clear`, history, templates |
| Test suite | 225 unit + integration tests | 322 unit + integration tests |
| Manual test docs | 4 docs, 138 tests | 5 docs, 303 tests |

## Decision

Archive `experimentation/llmforge/` in place. No code migration. The rustml
workspace crates are the authoritative implementation going forward.

- llmforge source is preserved read-only for historical reference.
- The `./llmf` launcher and its associated `bin/` scripts remain in place.
- No further feature development or maintenance will be done on llmforge.
- The llmforge manual test documents have been superseded by the production test
  docs in `docs/5-testing/`; see migration notes in
  `experimentation/llmforge/ARCHIVED.md`.

## Rationale

- **No dependency relationship**: llmforge was never wired to depend on rustml
  crates. A migration would require rewriting the llmforge codebase rather than
  simply re-pointing imports — the effort provides no value given the production
  crates already implement the same functionality at higher quality.
- **Feature parity exceeded**: the rustml workspace has surpassed llmforge on
  every axis (architecture support, test coverage, CLI ergonomics, multi-turn
  chat, performance profiling, SafeTensors multi-arch dispatch).
- **Historical value**: the llmforge codebase documents the original design
  exploration and the evolution of the LLM inference approach. Keeping it in
  `experimentation/` preserves that history without polluting the production
  workspace.

## Consequences

**Positive:**
- Clear single source of truth: all production LLM inference is in the rustml
  workspace crates.
- No confusion about which implementation to extend or test.
- llmforge test docs remain accessible for historical comparison.

**Negative:**
- llmforge's self-contained workspace (`experimentation/llmforge/Cargo.toml`)
  still builds independently. It will drift from rustml as the production crates
  evolve. This is accepted — llmforge is frozen.

## See Also

- `experimentation/llmforge/ARCHIVED.md` — archive marker with crate mapping
- `docs/5-testing/manual_testing.md` — production manual testing hub
- ADR-001 — Unified LlmModel for GPT-2 SafeTensors inference
