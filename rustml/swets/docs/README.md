# swets Documentation Hub

**Audience**: Developers, architects, contributors

## What

swets (SWE Time Series) is a from-scratch Rust ML framework for time series forecasting with tape-based autodiff.

## Why

rustml-core/nn/nlp focus on inference of pre-trained LLMs. swets adds training infrastructure: autodiff, optimizers, loss functions, and time series-specific data pipelines.

## How

Pure Rust, CPU-only, SIMD-accelerated. Extends rustml-core types (tensors, dtypes, storage) with a gradient tape and training loop.

## Where

| Phase | Path | Description |
|-------|------|-------------|
| Requirements | [1-requirements/](1-requirements/) | SRS |
| Design | [3-design/](3-design/) | Architecture |
| Development | [4-development/](4-development/) | Developer guide |
| Testing | [5-testing/](5-testing/) | Testing strategy, test plan |
