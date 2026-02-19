# Architecture

**Audience**: Developers, architects

## Overview

swets is a CPU-only Rust ML framework for time series forecasting. It extends rustml-core with tape-based reverse-mode autodiff and training infrastructure.

## Crate Structure

```
swets/
├── main/src/
│   ├── api/         # Public traits: Layer, Optimizer, Loss, BackwardOp
│   ├── core/        # Implementations
│   │   ├── ops/         # Forward + backward operations
│   │   ├── nn/          # Linear, LSTM, Conv1d, norms, activations
│   │   ├── optim/       # SGD, Adam, AdamW, LR schedulers
│   │   ├── loss/        # MSE, MAE, Huber, CrossEntropy, Quantile
│   │   ├── timeseries/  # OHLCV, dataset, features, scaler, dataloader
│   │   ├── models/      # TCN, Transformer, N-BEATS
│   │   ├── training/    # Trainer, metrics
│   │   └── serde/       # Checkpoint save/load
│   └── saf/         # Facade re-exports
└── tests/           # Integration tests
```

## Key Design Decisions

### Tape-Based Autodiff

Operations record onto a `GradientTape` during the forward pass. `backward()` replays in reverse, accumulating gradients in `HashMap<TensorId, Tensor>`. Tensors carry no gradient state; the tape owns all gradient information externally.

### Scoped `no_grad` Guard

`tape::no_grad(|| { ... })` temporarily disables tape recording so that forward passes inside the closure produce no tape entries. This is the standard mechanism for running inference (validation, prediction, serving) without the overhead of gradient tracking.

**Implementation**: `no_grad` saves the current `enabled` state before disabling the tape, then restores the previous state on exit rather than unconditionally re-enabling. This means nested calls stack correctly — an inner `no_grad` exiting will not re-enable recording while an outer `no_grad` scope is still active, matching PyTorch's `torch.no_grad()` semantics.

**Primary use site**: `Trainer::validate()` wraps its entire eval loop in `no_grad` to avoid polluting the tape between training steps.

### Shared Types with rustml-core

| Type | Shared |
|------|--------|
| `SmallVec<[usize; 4]>` shapes | Yes |
| `Arc<Storage>` data | Yes |
| `DType` enum | Yes |
| `GradientTape` | New in swets |

### Layer Trait

All layers take `&mut self` for stateful layers (BatchNorm, Dropout). The optional `tape: Option<&mut GradientTape>` parameter controls gradient tracking.

### Model Architectures

| Model | Use Case |
|-------|----------|
| TCN | Causal dilated convolutions for sequential forecasting |
| Transformer | Attention-based time series with positional encoding |
| LSTM | Recurrent sequential modeling |
| N-BEATS | Doubly-residual decomposition forecasting |

## Requirements Reference

See [SRS](../1-requirements/srs.md) for full functional and non-functional requirements.
