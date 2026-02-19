# Developer Guide

**Audience**: Contributors, developers

## Prerequisites

- Rust stable toolchain
- Cargo

## Build

```bash
cargo build --release
```

## Test

```bash
cargo test
```

## Project Layout

swets follows the rustml workspace conventions:

- `main/src/` — Source code with SEA layering (api/, core/, saf/)
- `tests/` — Integration tests
- `docs/` — Project documentation

## Adding a New Layer

1. Implement the `Layer` trait in `core/nn/`
2. Implement `BackwardOp` for any new differentiable operations
3. Add gradient correctness test in `tests/grad_check_test.rs`
4. Re-export from `saf/mod.rs`

## Adding a New Optimizer

1. Implement the `Optimizer` trait in `core/optim/`
2. Add convergence test in `tests/optim_test.rs`
3. Re-export from `saf/mod.rs`

## Conventions

- No `unsafe` outside SIMD kernels
- All backward ops use `_raw` (non-tracking) variants to prevent tape recursion
- Parameters returned by `Layer::parameters()` are positional; order must be stable
