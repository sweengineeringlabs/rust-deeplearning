# Testing Strategy

**Audience**: Developers, QA

## Approach

swets uses a layered testing strategy targeting correctness of gradient computation, layer behavior, optimizer convergence, and end-to-end training pipelines.

## Test Levels

### Unit Tests

- Tensor operations: creation, shape manipulation, arithmetic
- Individual backward op gradient correctness via finite-difference checks
- Feature engineering computations (RSI, moving average, volatility)
- Scaler transform/inverse_transform round-trip

### Integration Tests

- Layer forward/backward through tape
- Optimizer convergence on toy problems
- DataLoader iteration and batching
- Checkpoint save/load round-trip

### System Tests

- End-to-end training loop: dataset -> dataloader -> model -> loss -> backward -> optimizer step
- Model convergence on synthetic data (sine wave, linear trend)
- Causal masking verification (TCN, Transformer)

## Acceptance Criteria

| Criterion | Target |
|-----------|--------|
| Gradient check relative error (F32) | < 1e-4 |
| Training convergence (synthetic sine) | MSE < 0.01 in 100 epochs |
| Scaler round-trip error | < 1e-6 |
| Checkpoint round-trip | Bit-exact forward output |
