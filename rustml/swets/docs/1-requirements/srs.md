# Software Requirements Specification: swets

**Audience**: Developers, architects, contributors

**swets** (SWE Time Series) — A from-scratch Rust ML framework optimized for time series forecasting with tape-based autodiff.

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Status | Draft |
| Parent | `rustml` umbrella workspace |
| Design Ref | `experimentation/TimeSeriesML_framework_design.md` |

---

## 1. Purpose

swets extends the rustml ecosystem with **training infrastructure** for time series forecasting. Where rustml-core/nn/nlp focus on inference of pre-trained LLMs, swets provides:

1. A tape-based reverse-mode automatic differentiation engine
2. Neural network layers with gradient tracking
3. Optimizers, loss functions, and LR schedulers
4. Time series-specific data loading, feature engineering, and normalization
5. Pre-built model architectures (TCN, Transformer, LSTM, N-BEATS)
6. Training loop infrastructure with early stopping, checkpointing, and metrics

### 1.1 Scope

swets is a **CPU-only, single-machine** training framework. It targets small-to-medium models (< 100M parameters) on sequential/OHLCV data. GPU support and distributed training are explicitly out of scope for v1.

### 1.2 Relationship to rustml

swets reuses rustml-core's foundational types where possible:

| Shared with rustml-core | New in swets |
|------------------------|--------------|
| `SmallVec<[usize; 4]>` shapes | `GradientTape` (autodiff engine) |
| `Arc<Storage>` data buffers | `BackwardOp` trait (gradient ops) |
| `DType` enum | `Layer` trait (trainable modules) |
| `Device` enum | `Optimizer` trait |
| SIMD kernel patterns | Training loop, checkpointing |

---

## 2. Functional Requirements

### 2.1 Core Tensor System

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-100 | Tensor struct with unique `TensorId`, `Arc<Storage>` data, shape, strides, dtype, device | Must |
| FR-101 | `TensorId` generated via atomic counter for tape tracking | Must |
| FR-102 | `Storage` enum: `Owned(Vec<u8>)`, `View { parent, offset, len }`, `MMap` | Must |
| FR-103 | `DType` enum matching rustml-core: F32, F16, BF16, I8, U8, Q8_0, Q4_0, Q4_1 | Must |
| FR-104 | Thread-local `TensorPool` for buffer recycling during backward pass | Should |
| FR-105 | Creation ops: `zeros`, `ones`, `randn`, `uniform`, `from_vec`, `arange` | Must |
| FR-106 | Shape ops: `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `flatten`, `view` | Must |
| FR-107 | Indexing ops: `slice`, `index_select`, `gather`, `masked_select` | Must |
| FR-108 | Element-wise math: add, sub, mul, div, pow, sqrt, exp, log, abs, neg | Must |
| FR-109 | Reductions: sum, mean, max, min, std, var | Must |
| FR-110 | `matmul` and `dot` with broadcasting | Must |
| FR-111 | `conv1d` with stride, padding, and dilation parameters | Must |
| FR-112 | `concat`, `stack`, `split` tensor joining/splitting | Must |
| FR-113 | Row-major strides computation and flat offset indexing | Must |

### 2.2 Automatic Differentiation

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-200 | `GradientTape` records forward operations as `TapeEntry` list | Must |
| FR-201 | `TapeEntry` stores: `BackwardOp`, output ID, input IDs, saved tensors | Must |
| FR-202 | `BackwardOp` trait with `backward(grad_output, saved) -> Vec<input_grads>` | Must |
| FR-203 | `GradientTape::backward(loss_id)` replays tape in reverse, accumulates gradients in `HashMap<TensorId, Tensor>` | Must |
| FR-204 | `GradientTape::grad(id)` retrieves gradient for a tensor | Must |
| FR-205 | `GradientTape::clear()` resets ops and gradients between training steps | Must |
| FR-206 | `tape.enabled` flag to disable recording (inference mode) | Must |
| FR-207 | `Layer::forward` accepts `Option<&mut GradientTape>` — `None` means no recording | Must |
| FR-208 | Scoped `no_grad` helper that temporarily disables tape | Should |
| FR-209 | Non-tracking `_raw` variants of ops (e.g. `matmul_raw`) for use inside backward implementations | Must |

#### Backward Op Coverage

| ID | Operation | Gradient Formula | Priority |
|----|-----------|------------------|----------|
| FR-210 | MatMul | `grad_a = grad @ b^T`, `grad_b = a^T @ grad` | Must |
| FR-211 | Add | `unbroadcast(grad)` per input | Must |
| FR-212 | Mul (element-wise) | `grad * other_input` | Must |
| FR-213 | Sigmoid | `grad * sigmoid(x) * (1 - sigmoid(x))` | Must |
| FR-214 | Tanh | `grad * (1 - tanh(x)^2)` | Must |
| FR-215 | ReLU | `grad * (input > 0)` | Must |
| FR-216 | Softmax | `s * (g - sum(g * s, dim))` | Must |
| FR-217 | Conv1d | Transposed convolution for input grad, cross-correlation for weight grad | Must |

### 2.3 Neural Network Layers

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-300 | `Layer` trait: `forward(&mut self, input, tape)`, `parameters()`, `parameters_mut()`, `train()`, `eval()`, `parameter_count()` | Must |
| FR-301 | `Sequential` container chaining arbitrary layers | Must |
| FR-302 | `Linear` layer with Xavier (default) and He initialization | Must |
| FR-303 | `LSTM` layer: multi-layer, configurable hidden size, optional bidirectional, dropout between layers | Must |
| FR-304 | `Conv1d` layer with stride, padding, dilation | Must |
| FR-305 | Activation layers: `ReLU`, `Tanh`, `Sigmoid`, `GELU`, `SiLU` | Must |
| FR-306 | `LayerNorm` with configurable epsilon | Must |
| FR-307 | `BatchNorm1d` with running statistics (updated via `&mut self` in training mode) | Should |
| FR-308 | `Dropout` layer: active in training, identity in eval | Must |
| FR-309 | `model_summary()` prints total/trainable/frozen parameter counts | Should |

### 2.4 Optimizers

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-400 | `Optimizer` trait: `step(params, tape)`, `lr()`, `set_lr()` — reads gradients from tape | Must |
| FR-401 | `SGD` with momentum, dampening, weight decay, Nesterov | Must |
| FR-402 | `Adam` with configurable betas, epsilon, L2 regularization | Must |
| FR-403 | `AdamW` with decoupled weight decay (applied before Adam update) | Must |
| FR-404 | Gradient clipping by global L2 norm (`clip_grad_norm`) | Must |
| FR-405 | Gradient clipping by value (`clip_grad_value`) | Should |

### 2.5 Learning Rate Schedulers

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-500 | `LRScheduler` trait: `step(optimizer)`, `get_lr()` | Must |
| FR-501 | `StepLR` — decay by gamma every N steps | Must |
| FR-502 | `CosineAnnealingLR` — cosine decay to eta_min | Must |
| FR-503 | `WarmupCosineScheduler` — linear warmup then cosine decay | Must |

### 2.6 Loss Functions

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-600 | `Loss` trait: `forward(predictions, targets, tape)` | Must |
| FR-601 | `MSELoss` — mean squared error | Must |
| FR-602 | `MAELoss` — mean absolute error | Must |
| FR-603 | `HuberLoss` — quadratic near zero, linear far from zero, configurable delta | Must |
| FR-604 | `CrossEntropyLoss` — numerically stable log-softmax + NLL | Should |
| FR-605 | `QuantileLoss` — asymmetric loss for quantile regression | Should |

### 2.7 Time Series Data Pipeline

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-700 | `OHLCVCandle` struct: timestamp, open, high, low, close, volume | Must |
| FR-701 | `TimeSeriesDataset`: windowed access over candle data, configurable window size and prediction horizon | Must |
| FR-702 | Configurable target columns: single, multi, or close (default) | Must |
| FR-703 | `DataLoader` implementing `Iterator` — yields `(input_batch, target_batch)` | Must |
| FR-704 | `DataLoader` shuffle support with epoch reset | Must |
| FR-705 | `FeatureEngineer` with pluggable `Feature` trait | Must |
| FR-706 | Built-in features: log returns, moving average, volatility | Must |
| FR-707 | RSI feature using Wilder's exponential moving average (not SMA) | Must |
| FR-708 | `Scaler` with MinMax, StandardScaler, RobustScaler | Must |
| FR-709 | `Scaler::inverse_transform` for denormalization | Must |

### 2.8 Model Architectures

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-800 | **TCN** (Temporal Convolutional Network): causal padding, exponentially increasing dilation, residual connections, global average pooling | Must |
| FR-801 | **TimeSeriesTransformer**: input embedding, sinusoidal positional encoding, N transformer blocks with causal masking, last-timestep projection | Must |
| FR-802 | Multi-head attention with Q/K/V projections, scaled dot-product, causal + padding mask support | Must |
| FR-803 | **N-BEATS**: FC stacks with backcast subtraction (doubly-residual), forecast accumulation, configurable stacks/blocks/layers | Should |
| FR-804 | LSTM-based forecasting model (built from FR-303) | Must |

### 2.9 Training Infrastructure

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-900 | `Trainer` struct: owns model, optimizer, loss function | Must |
| FR-901 | `train_epoch()`: forward pass, backward pass, gradient clipping, optimizer step — fresh tape per batch | Must |
| FR-902 | `validate()`: forward pass with no tape (no gradient computation) | Must |
| FR-903 | `fit()`: epoch loop with train/validate, logging, scheduler step, early stopping check | Must |
| FR-904 | `predict()`: switches model to eval mode, runs forward with no tape | Must |
| FR-905 | Early stopping with configurable patience | Must |
| FR-906 | Best-model checkpointing on validation loss improvement | Must |
| FR-907 | Builder pattern: `with_scheduler()`, `with_early_stopping()`, `with_grad_clip()`, `with_checkpoint_dir()` | Should |

### 2.10 Serialization

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1000 | `Checkpoint` struct: model parameters (name, bytes, shape, dtype), optimizer state, epoch, best_val_loss | Must |
| FR-1001 | `save_checkpoint(model, path)` — serialize parameters via bincode | Must |
| FR-1002 | `load_checkpoint(model, path)` — deserialize and load into model | Must |

### 2.11 Metrics

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1100 | Accumulator-based `Metrics` struct with `update(pred, target)` and `reset()` | Must |
| FR-1101 | MSE, MAE, RMSE | Must |
| FR-1102 | R-squared (coefficient of determination) | Must |
| FR-1103 | MAPE and SMAPE | Should |

---

## 3. Non-Functional Requirements

### 3.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-100 | SIMD-accelerated tensor operations (matmul, conv1d, element-wise) | Comparable to single-threaded C |
| NFR-101 | Cache-friendly memory layout (row-major, contiguous where possible) | L1/L2 utilization > 80% |
| NFR-102 | Buffer recycling via `TensorPool` reduces allocation pressure in backward pass | < 10% time in alloc during training |
| NFR-103 | Zero-copy views for reshape/transpose/slice | No allocation for shape-only ops |

### 3.2 Correctness

| ID | Requirement |
|----|-------------|
| NFR-200 | Gradient correctness validated via finite-difference numerical checks for all BackwardOp implementations |
| NFR-201 | Scaler handles constant features (zero variance/range) without producing NaN/Inf |
| NFR-202 | CrossEntropyLoss uses numerically stable log-softmax (subtract max before exp) |
| NFR-203 | Causal masking prevents information leakage from future timesteps |
| NFR-204 | RSI uses Wilder's EMA (smoothing factor 1/period), not simple moving average |

### 3.3 Rust Safety

| ID | Requirement |
|----|-------------|
| NFR-300 | No `unsafe` code outside SIMD kernels |
| NFR-301 | `Storage::View` holds `Arc<Storage>` (not `Arc<Tensor>`) to prevent reference cycles |
| NFR-302 | `GradientTape` borrow semantics must compile — design must resolve `Option<&mut GradientTape>` threading through nested layer calls (see Section 5.1) |
| NFR-303 | All `BackwardOp` implementations must be `Send + Sync` |

### 3.4 Compatibility

| ID | Requirement |
|----|-------------|
| NFR-400 | `DType` enum identical to rustml-core for weight interop |
| NFR-401 | `Shape` type alias matches rustml-core (`SmallVec<[usize; 4]>`) |
| NFR-402 | Project follows rustml workspace conventions: `main/src/` layout, SEA layering, umbrella pattern |
| NFR-403 | Minimum supported Rust version: stable (no nightly-only features outside optional SIMD) |

### 3.5 Extensibility

| ID | Requirement |
|----|-------------|
| NFR-500 | New `BackwardOp` implementations can be added without modifying the tape |
| NFR-501 | New `Layer` implementations can be used with existing `Sequential`, `Trainer`, `Optimizer` |
| NFR-502 | New `Feature` implementations can be plugged into `FeatureEngineer` |
| NFR-503 | New `Loss`, `Optimizer`, `LRScheduler` implementations via traits |

---

## 4. Constraints

| ID | Constraint |
|----|-----------|
| CON-01 | CPU only — no GPU/CUDA/Metal support in v1 |
| CON-02 | Single-machine — no distributed training |
| CON-03 | No feature parity with PyTorch — focused subset for time series |
| CON-04 | Pure Rust with minimal external dependencies (smallvec, memmap2, serde, bincode, rand) |
| CON-05 | Educational but production-capable — code clarity is a first-order concern |

---

## 5. Known Design Risks

Issues identified during design review that must be resolved before or during implementation.

### 5.1 Tape Borrow Semantics (Critical)

**Problem:** The design passes `Option<&mut GradientTape>` through nested layer calls. In Rust, you cannot reborrow `&mut T` through `Option::as_deref_mut()` in a loop, nor pass the same `&mut` to multiple callees within one function body. This affects `Sequential::forward`, `LSTM::forward`/`lstm_cell`, and all compound layers.

**Impact:** Code as designed will not compile.

**Mitigation options:**
1. Interior mutability: `Rc<RefCell<GradientTape>>` or `Arc<Mutex<GradientTape>>`
2. Thread-local tape: global `thread_local!` tape, toggled on/off
3. Explicit reborrowing protocol: restructure signatures to take `&mut Option<&mut GradientTape>` and reborrow at each call site

### 5.2 Positional Parameter Serialization

**Problem:** Parameters are serialized as `param_0, param_1, ...` by position. If the model structure changes between save and load, parameters silently mismatch.

**Mitigation:** Add named parameter support to the `Layer` trait (e.g., `fn named_parameters(&self) -> Vec<(&str, &Tensor)>`).

### 5.3 Scaler Division by Zero

**Problem:** `Scaler::transform` divides by `max - min`, `std`, or IQR, none of which are guarded against zero for constant features.

**Mitigation:** Clamp denominator to `max(value, epsilon)` where epsilon is a small positive constant (e.g., 1e-8).

### 5.4 Optimizer Step Type Mismatch

**Problem:** `Trainer::train_epoch` collects `Vec<&mut Tensor>` then passes `&mut params.iter_mut().collect::<Vec<_>>()` to `optimizer.step()`, producing `&mut [&mut &mut Tensor]` instead of `&mut [&mut Tensor]`.

**Mitigation:** Pass `&mut params[..]` directly to `optimizer.step()`.

### 5.5 LSTM Not Composable via Layer Trait

**Problem:** `LSTM::forward` has a custom signature accepting `state: Option<(Tensor, Tensor)>`, so it cannot implement the `Layer` trait or be used in `Sequential`.

**Mitigation:** Either add a `StatefulLayer` trait, or have LSTM manage state internally with a `reset_state()` method.

---

## 6. Project Structure

Following rustml workspace conventions:

```
rustml/swets/
├── Cargo.toml              # Package (member of rustml workspace)
├── docs/
│   └── 1-requirements/
│       └── srs.md          # This document
├── main/
│   └── src/
│       ├── lib.rs
│       ├── api/            # Public types, traits, error types
│       │   ├── mod.rs
│       │   ├── tensor.rs   # Tensor, TensorId, Shape, DType, Storage
│       │   ├── tape.rs     # GradientTape, BackwardOp, TapeEntry
│       │   ├── layer.rs    # Layer trait, Sequential
│       │   ├── optim.rs    # Optimizer, LRScheduler traits
│       │   └── loss.rs     # Loss trait
│       ├── core/           # Implementation
│       │   ├── mod.rs
│       │   ├── ops/        # Forward + backward op implementations
│       │   ├── nn/         # Linear, LSTM, Conv1d, activations, norms, dropout
│       │   ├── optim/      # SGD, Adam, AdamW, schedulers, grad clipping
│       │   ├── loss/       # MSE, MAE, Huber, CrossEntropy, Quantile
│       │   ├── timeseries/ # OHLCVCandle, dataset, features, scaler, dataloader
│       │   ├── models/     # TCN, Transformer, N-BEATS
│       │   ├── training/   # Trainer, metrics
│       │   ├── serde/      # Checkpoint, save/load
│       │   └── pool.rs     # TensorPool
│       └── saf/            # Facade re-exports
│           └── mod.rs
└── tests/
    ├── grad_check_test.rs      # Numerical gradient verification
    ├── layer_test.rs           # Layer forward/backward
    ├── optim_test.rs           # Optimizer convergence
    ├── training_test.rs        # End-to-end training loop
    └── timeseries_test.rs      # Data pipeline, features, scaler
```

---

## 7. Acceptance Criteria

### 7.1 Gradient Correctness

All `BackwardOp` implementations pass finite-difference gradient checks with relative error < 1e-4 for F32.

### 7.2 Training Convergence

A 2-layer LSTM or small TCN trained on synthetic sine wave data converges to MSE < 0.01 within 100 epochs.

### 7.3 Scaler Round-Trip

For all scaler types: `inverse_transform(transform(data)) == data` within floating-point tolerance (< 1e-6).

### 7.4 Causal Masking

TCN and Transformer outputs at timestep `t` are independent of inputs at timesteps `> t`, verified by zeroing future inputs and confirming identical output.

### 7.5 Checkpoint Round-Trip

`save_checkpoint` followed by `load_checkpoint` produces a model that yields identical forward pass output on the same input.

### 7.6 Compilation

All code compiles on stable Rust with no `unsafe` outside SIMD kernels. The tape borrow semantics (Section 5.1) must be resolved to a pattern that compiles and passes all tests.
