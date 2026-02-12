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
├── rustml/                     # Umbrella (4 sub-crates)
│   ├── core/                   # rustml-core
│   ├── nn/                     # rustml-nn
│   ├── hub/                    # rustml-hub
│   └── nlp/                    # rustml-nlp
├── audiolearn/                 # Umbrella (2 sub-crates)
│   ├── app/                    # audiolearn-app
│   └── tauri/                  # audiolearn-tauri
├── components/                 # Single-crate
└── tutorial-app/               # Single-crate
```
