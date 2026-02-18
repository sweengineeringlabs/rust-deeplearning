# Project Structure

## Layout Conventions

### 1. `{main, tests}` Layout

Every package uses the `main/src/` layout for source code and `tests/` for integration tests:

```
crate-name/
├── Cargo.toml
├── main/
│   └── src/
│       ├── lib.rs
│       └── ...
└── tests/
    └── crate_name_int_test.rs
```

### 2. SEA Layering

Every library crate follows SEA (Stratified Encapsulation Architecture) inside `main/src/`:

```
main/src/
├── lib.rs          # pub mod api; mod core; mod saf; pub use saf::*;
├── api/            # Public types, traits, error types
├── core/           # Implementation details
├── saf/            # Thin re-exports (facade)
└── spi/            # (optional) Service Provider Interfaces
```

- **api/** — Public contracts: types, traits, error enums
- **core/** — Business logic, hidden behind `mod core;` (not `pub mod`)
- **saf/** — Facade re-exports, factory functions
- **spi/** — Only when pluggable backends are needed

### 3. Umbrella Pattern

Multi-crate features use a directory with a `[workspace]`-only `Cargo.toml`:

```
umbrella/
├── Cargo.toml      # [workspace] only — not a package
├── sub-a/
│   ├── Cargo.toml
│   └── main/src/...
└── sub-b/
    ├── Cargo.toml
    └── main/src/...
```

### 4. No `*-common` Crates

Types belong to the interface they define (`api/` or `spi/`), not a shared `common` module.

### 5. Naming

- Kebab-case folder names
- Kebab-case crate names
- `docs/` at project level, `doc/` optional per-crate

## Current Structure

```
rust-deeplearning/
├── Cargo.toml                  # Root workspace
├── docs/                       # Project-level docs
│   └── 3-design/adr/           # Architecture Decision Records
├── rustml/                     # Umbrella (4 sub-crates + CLI + extras)
│   ├── core/                   # rustml-core (tensors, dtypes)
│   ├── nn/                     # rustml-nn (layers, KVCache, attention)
│   ├── hub/                    # rustml-hub (HuggingFace downloads, SafeTensors)
│   ├── nlp/                    # rustml-nlp (models, generation, tokenizer bridge)
│   ├── gguf/                   # rustml-gguf (GGUF parsing, weight loading)
│   ├── tokenizer/              # rustml-tokenizer (BPE, GGUF, byte tokenizers)
│   └── cli/                    # rustml-cli (sweai unified binary)
├── audiolearn/                 # Umbrella (2 sub-crates)
│   ├── app/                    # audiolearn-app
│   └── tauri/                  # audiolearn-tauri
├── components/                 # Single-crate
└── tutorial-app/               # Single-crate
```

### Model Architecture

The `rustml-nlp` crate has two model implementations:

| Model | Location | Purpose | KV Cache | Used by CLI |
|-------|----------|---------|----------|-------------|
| **`LlmModel`** | `nlp/main/src/core/model.rs` | Unified production model — supports GPT-2, Llama, Gemma, Falcon, Mixtral via config | Yes | Yes (GGUF + SafeTensors) |
| **`GptModel`** | `nlp/main/src/core/gpt.rs` | Standalone GPT-2 reference implementation for learning | No (O(n^2)) | No (teaching only) |

Both GGUF and SafeTensors inference routes use `LlmModel`. See [ADR-001](adr/adr-001-unified-llmmodel-for-gpt2.md).

### Hub Authentication

The `rustml-hub` crate (`HubApi`) resolves HuggingFace API tokens with the following precedence:

| Priority | Source | How to set |
|----------|--------|------------|
| 1 (highest) | `--token` CLI flag | `sweai hub --token hf_xxx download ...` or `rustml-hub-cli --token hf_xxx download ...` |
| 2 | `HF_TOKEN` environment variable | `export HF_TOKEN=hf_xxx` |
| 3 (lowest) | Token file | `~/.cache/huggingface/token` (written by `huggingface-cli login`) |

**How it works:**

- `HubApi::new()` reads `HF_TOKEN` from the environment (priority 2)
- `HubApi::with_token()` overrides with an explicit token (priority 1, used by `--token` flag)
- The `hf-hub` crate (v0.4.x) reads the token file automatically when no programmatic token is supplied (priority 3)
- The async download path (`download_model`) sends the token via `Authorization: Bearer` header
- The sync download path (`download_model_sync`, `download_gguf_sync`) passes the token to `hf_hub::api::sync::ApiBuilder::with_token()`

**Note:** The `hf-hub` crate v0.4.x does **not** read `HF_TOKEN` from the environment on its own — `HubApi` bridges that gap. Gated models (e.g., `google/gemma-3-1b-it`) require a token with accepted license terms on HuggingFace.
