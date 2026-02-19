# TimeSeriesML Framework Design
## A from-scratch Rust ML framework optimized for time series forecasting

---

## 1. Design Philosophy & Architecture

**Goals:**
1. Pure Rust implementation with minimal dependencies
2. Optimized for CPU performance (SIMD, cache-friendly)
3. Time series first-class citizen (OHLCV, sequential data)
4. Type-safe, compile-time shape checking where possible
5. Educational but production-capable
6. Modular architecture for extensibility

**Non-Goals:**
- GPU support (initially — can be added later)
- Distributed training
- Complete feature parity with PyTorch

**Relationship to rustml-core:** This framework extends rustml with training infrastructure.
The core tensor system aligns with rustml-core patterns: `SmallVec<[usize; 4]>` shapes,
`Arc<Storage>` data, and the same `DType` enum. The key addition is a **tape-based
reverse-mode autodiff** engine that enables gradient computation for training, something
rustml-core's inference-only design does not provide.

```
timeseriesml/
├── core/           # Tensor operations and autodiff (tape-based)
├── nn/             # Neural network layers
├── optim/          # Optimizers and gradient clipping
├── data/           # Data loading and preprocessing
├── timeseries/     # Time series specific components
├── models/         # Pre-built model architectures
├── training/       # Training infrastructure
└── serde/          # Serialization and checkpointing
```

---

## 2. Core Tensor System

### Tensor Structure

The tensor struct carries **no gradient state**. Gradients live on the `GradientTape`
(Section 3), not on the tensor itself. Each tensor has a unique `id` so the tape can
track which tensors participated in which operations.

```rust
use smallvec::SmallVec;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonically increasing ID generator for tensors.
static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(0);

/// Unique identifier for a tensor, used by GradientTape to track gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(u64);

impl TensorId {
    fn next() -> Self {
        Self(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Stack-allocated for ≤4 dims, matching rustml-core TensorShape.
pub type Shape = SmallVec<[usize; 4]>;

pub struct Tensor {
    /// Unique identifier for tape-based gradient tracking.
    pub id: TensorId,

    /// Data storage — reference-counted for cheap clones.
    data: Arc<Storage>,

    /// Shape (e.g. [batch, seq_len, features]).
    shape: Shape,

    /// Strides for index computation.
    strides: Shape,

    /// Element data type.
    dtype: DType,

    /// Device (CPU only for now).
    device: Device,
}
```

> **Note:** There is no `grad`, `requires_grad`, or `grad_fn` field. The tape (Section 3)
> owns all gradient information externally.

### Storage

`Storage::View` holds `Arc<Storage>` — **not** `Arc<Tensor>` — to prevent reference
cycles. A view shares the parent's raw buffer without creating a circular dependency.

```rust
pub enum Storage {
    /// Owned contiguous buffer.
    Owned(Vec<u8>),

    /// Zero-copy view into another storage's buffer.
    View {
        parent: Arc<Storage>,
        offset: usize,
        len: usize,
    },

    /// Memory-mapped file region.
    MMap {
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        len: usize,
    },
}
```

### DType

Matches rustml-core's enum for interoperability:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    #[default]
    F32,
    F16,
    BF16,
    I8,
    U8,
    /// Block-quantized 8-bit: 32 elements/block, 34 bytes/block.
    Q8_0,
    /// Block-quantized 4-bit: 32 elements/block, 18 bytes/block.
    Q4_0,
    /// Block-quantized 4-bit with min: 32 elements/block, 20 bytes/block.
    Q4_1,
}

impl DType {
    /// Per-element byte size. Returns 0 for block-quantized types.
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
            DType::Q8_0 | DType::Q4_0 | DType::Q4_1 => 0,
        }
    }
}

pub enum Device {
    Cpu,
}
```

### TensorPool — Buffer Recycling

To reduce allocation pressure during training, a thread-local pool recycles
previously-allocated buffers. The tape's `backward()` pass benefits most since it
creates many intermediate gradient tensors.

```rust
use std::cell::RefCell;

thread_local! {
    static POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new());
}

pub struct TensorPool {
    /// Buckets keyed by byte length (rounded up to power-of-two).
    buckets: HashMap<usize, Vec<Vec<u8>>>,
    max_cached: usize,
}

impl TensorPool {
    pub fn new() -> Self {
        Self { buckets: HashMap::new(), max_cached: 64 }
    }

    /// Get a buffer of at least `len` bytes, reusing a cached one if available.
    pub fn get(&mut self, len: usize) -> Vec<u8> {
        let key = len.next_power_of_two();
        self.buckets.get_mut(&key)
            .and_then(|v| v.pop())
            .unwrap_or_else(|| vec![0u8; key])
    }

    /// Return a buffer to the pool for reuse.
    pub fn recycle(&mut self, mut buf: Vec<u8>) {
        let key = buf.capacity().next_power_of_two();
        let bucket = self.buckets.entry(key).or_default();
        if bucket.len() < self.max_cached {
            buf.clear();
            bucket.push(buf);
        }
        // else: drop buf normally
    }
}

/// Convenience: allocate a pooled tensor buffer.
pub fn pooled_buffer(byte_len: usize) -> Vec<u8> {
    POOL.with(|p| p.borrow_mut().get(byte_len))
}

/// Convenience: return a buffer to the pool.
pub fn recycle_buffer(buf: Vec<u8>) {
    POOL.with(|p| p.borrow_mut().recycle(buf));
}
```

### Core Operations

**Creation Operations:**
- `Tensor::zeros(shape)` — Create tensor filled with zeros
- `Tensor::ones(shape)` — Create tensor filled with ones
- `Tensor::randn(shape)` — Random normal distribution
- `Tensor::uniform(shape, low, high)` — Uniform random
- `Tensor::from_vec(data, shape)` — From raw data
- `Tensor::arange(start, end, step)` — Range tensor

**Shape Operations:**
- `reshape(new_shape)` — Change shape (view if possible)
- `transpose(dim0, dim1)` — Swap dimensions
- `permute(dims)` — Reorder dimensions
- `squeeze(dim)` / `unsqueeze(dim)` — Add/remove size-1 dims
- `flatten()` — Flatten to 1D
- `view(shape)` — Zero-copy reshape

**Indexing & Slicing:**
- `slice(ranges)` — Multi-dimensional slicing
- `index_select(dim, indices)` — Select along dimension
- `gather(dim, indices)` — Gather values
- `masked_select(mask)` — Boolean indexing

**Math Operations:**
- Element-wise: `add, sub, mul, div, pow, sqrt, exp, log, abs, neg`
- Reduction: `sum, mean, max, min, std, var`
- `matmul(other)` — Matrix multiplication
- `dot(other)` — Dot product
- Broadcasting support for all binary operations

**Advanced Operations:**
- `conv1d(weight, bias, stride, padding, dilation)` — 1D convolution with explicit dilation
- `concat(tensors, dim)` / `stack(tensors, dim)` — Join tensors
- `split(sizes, dim)` — Split tensor

**Einsum (design note):** Full einsum will follow a contraction-path approach — parse the
subscript string into a sequence of pairwise contractions, each implemented as a
transpose + matmul. For v1, we provide the most common patterns as named ops (batched
matmul, outer product, trace) and defer general einsum to a later milestone.

### Memory Layout Helpers

```rust
/// Compute row-major strides from shape.
fn compute_strides(shape: &[usize]) -> Shape {
    if shape.is_empty() {
        return SmallVec::new();
    }
    let mut strides = smallvec::smallvec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Flat offset from multi-dimensional indices.
fn compute_offset(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter()
        .zip(strides.iter())
        .map(|(i, s)| i * s)
        .sum()
}
```

---

## 3. Automatic Differentiation — Tape-Based Reverse Mode

The autodiff engine uses a **gradient tape** rather than a per-tensor computational graph.
Operations record themselves onto the tape during the forward pass. Calling `backward()`
replays the tape in reverse, accumulating gradients in a `HashMap<TensorId, Tensor>` owned
by the tape.

### GradientTape

```rust
use std::collections::HashMap;

pub struct GradientTape {
    /// Recorded operations, in forward-execution order.
    ops: Vec<TapeEntry>,

    /// Accumulated gradients, keyed by tensor ID.
    grads: HashMap<TensorId, Tensor>,

    /// When false, operations are not recorded (inference / no_grad mode).
    pub enabled: bool,
}

struct TapeEntry {
    /// The backward operation to execute.
    op: Box<dyn BackwardOp>,

    /// ID of the output tensor produced by this operation.
    output_id: TensorId,

    /// IDs of the input tensors (whose gradients we accumulate).
    input_ids: SmallVec<[TensorId; 2]>,

    /// Tensors saved during the forward pass (e.g. inputs needed for backward).
    /// Owned by the tape entry — not by the tensor itself.
    saved_tensors: SmallVec<[Tensor; 2]>,
}
```

### BackwardOp Trait

Each differentiable operation implements this trait. The `backward` method receives the
upstream gradient and the saved tensors, and returns gradients for each input.

```rust
pub trait BackwardOp: Send + Sync {
    /// Compute gradients for each input.
    ///
    /// `grad_output`: gradient flowing back from downstream.
    /// `saved`: tensors saved during the forward pass (owned by TapeEntry).
    ///
    /// Returns one gradient per input, in the same order as `input_ids`.
    fn backward(
        &self,
        grad_output: &Tensor,
        saved: &[Tensor],
    ) -> SmallVec<[Tensor; 2]>;
}
```

### Recording and Backward

```rust
impl GradientTape {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            grads: HashMap::new(),
            enabled: true,
        }
    }

    /// Record a forward operation on the tape.
    pub fn record(
        &mut self,
        op: Box<dyn BackwardOp>,
        output_id: TensorId,
        input_ids: SmallVec<[TensorId; 2]>,
        saved_tensors: SmallVec<[Tensor; 2]>,
    ) {
        if self.enabled {
            self.ops.push(TapeEntry {
                op,
                output_id,
                input_ids,
                saved_tensors,
            });
        }
    }

    /// Run reverse-mode autodiff from `loss_id`.
    ///
    /// Seeds the loss gradient as ones, then iterates the tape in reverse,
    /// accumulating gradients in `self.grads`.
    pub fn backward(&mut self, loss_id: TensorId, loss_shape: &[usize]) {
        // Seed: dL/dL = 1
        self.grads.insert(loss_id, Tensor::ones(loss_shape));

        // Reverse iterate
        for entry in self.ops.iter().rev() {
            let grad_output = match self.grads.get(&entry.output_id) {
                Some(g) => g.clone(),
                None => continue, // no gradient flows to this op
            };

            let input_grads = entry.op.backward(&grad_output, &entry.saved_tensors);

            // Accumulate gradients for each input
            for (id, grad) in entry.input_ids.iter().zip(input_grads.into_iter()) {
                self.grads.entry(*id)
                    .and_modify(|existing| *existing = existing.add_raw(&grad))
                    .or_insert(grad);
            }
        }
    }

    /// Retrieve the gradient for a given tensor. Returns None if the tensor
    /// did not participate in the computation or has no gradient.
    pub fn grad(&self, id: TensorId) -> Option<&Tensor> {
        self.grads.get(&id)
    }

    /// Clear all recorded ops and gradients (call between training steps).
    pub fn clear(&mut self) {
        self.ops.clear();
        self.grads.clear();
    }
}
```

### No-Grad / Inference Mode

There are two ways to disable gradient tracking:

1. **Set `tape.enabled = false`:** Operations still accept `&mut GradientTape` but
   skip recording.
2. **Pass `None` as the tape:** The `Layer::forward` signature accepts
   `Option<&mut GradientTape>`, so passing `None` means no recording at all.

```rust
/// Scoped no-grad helper.
pub fn no_grad<F, R>(tape: &mut GradientTape, f: F) -> R
where
    F: FnOnce(&mut GradientTape) -> R,
{
    let was_enabled = tape.enabled;
    tape.enabled = false;
    let result = f(tape);
    tape.enabled = was_enabled;
    result
}
```

### BackwardOp Implementations

#### MatMul

The backward pass must not recurse through the tape. We use `matmul_raw`, a
non-tracking variant that performs the multiplication without recording to any tape.

```rust
/// Non-tracking matrix multiply — used inside backward ops to avoid
/// infinite recursion through the tape.
pub fn matmul_raw(a: &Tensor, b: &Tensor) -> Tensor {
    // Same SIMD/tiled implementation as forward matmul, but
    // no tape interaction whatsoever.
    matmul_kernel(a, b)
}

struct MatMulBackward;

impl BackwardOp for MatMulBackward {
    fn backward(
        &self,
        grad_output: &Tensor,
        saved: &[Tensor],
    ) -> SmallVec<[Tensor; 2]> {
        let a = &saved[0]; // [M, K]
        let b = &saved[1]; // [K, N]

        // grad_a = grad_output @ b^T     [M, N] @ [N, K] = [M, K]
        let grad_a = matmul_raw(grad_output, &b.transpose(-1, -2));
        // grad_b = a^T @ grad_output      [K, M] @ [M, N] = [K, N]
        let grad_b = matmul_raw(&a.transpose(-1, -2), grad_output);

        smallvec::smallvec![grad_a, grad_b]
    }
}

/// Forward matmul that records to tape.
pub fn matmul(
    a: &Tensor,
    b: &Tensor,
    tape: Option<&mut GradientTape>,
) -> Tensor {
    let result = matmul_raw(a, b);

    if let Some(tape) = tape {
        tape.record(
            Box::new(MatMulBackward),
            result.id,
            smallvec::smallvec![a.id, b.id],
            smallvec::smallvec![a.clone(), b.clone()],
        );
    }

    result
}
```

#### Add

```rust
struct AddBackward {
    left_shape: Shape,
    right_shape: Shape,
}

impl BackwardOp for AddBackward {
    fn backward(
        &self,
        grad_output: &Tensor,
        _saved: &[Tensor],
    ) -> SmallVec<[Tensor; 2]> {
        smallvec::smallvec![
            unbroadcast(grad_output, &self.left_shape),
            unbroadcast(grad_output, &self.right_shape),
        ]
    }
}
```

#### Mul (element-wise)

```rust
struct MulBackward;

impl BackwardOp for MulBackward {
    fn backward(
        &self,
        grad_output: &Tensor,
        saved: &[Tensor],
    ) -> SmallVec<[Tensor; 2]> {
        let a = &saved[0];
        let b = &saved[1];
        smallvec::smallvec![
            mul_raw(grad_output, b),  // d/da(a*b) = b
            mul_raw(grad_output, a),  // d/db(a*b) = a
        ]
    }
}
```

#### Sigmoid

```rust
struct SigmoidBackward;

impl BackwardOp for SigmoidBackward {
    fn backward(
        &self,
        grad_output: &Tensor,
        saved: &[Tensor],
    ) -> SmallVec<[Tensor; 2]> {
        let output = &saved[0]; // sigmoid(x), saved from forward
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        let grad = mul_raw(
            grad_output,
            &mul_raw(output, &sub_raw(&Tensor::ones_like(output), output)),
        );
        smallvec::smallvec![grad]
    }
}
```

#### Tanh

```rust
struct TanhBackward;

impl BackwardOp for TanhBackward {
    fn backward(
        &self,
        grad_output: &Tensor,
        saved: &[Tensor],
    ) -> SmallVec<[Tensor; 2]> {
        let output = &saved[0]; // tanh(x)
        // d/dx tanh(x) = 1 - tanh(x)^2
        let grad = mul_raw(
            grad_output,
            &sub_raw(&Tensor::ones_like(output), &mul_raw(output, output)),
        );
        smallvec::smallvec![grad]
    }
}
```

#### ReLU

```rust
struct ReLUBackward;

impl BackwardOp for ReLUBackward {
    fn backward(
        &self,
        grad_output: &Tensor,
        saved: &[Tensor],
    ) -> SmallVec<[Tensor; 2]> {
        let input = &saved[0];
        // grad * (input > 0)
        let mask = input.greater_than_scalar(0.0);
        smallvec::smallvec![mul_raw(grad_output, &mask)]
    }
}
```

#### Softmax

```rust
struct SoftmaxBackward {
    dim: i32,
}

impl BackwardOp for SoftmaxBackward {
    fn backward(
        &self,
        grad_output: &Tensor,
        saved: &[Tensor],
    ) -> SmallVec<[Tensor; 2]> {
        let output = &saved[0]; // softmax(x)
        // Jacobian-vector product: s * (g - sum(g * s, dim))
        let sum_gs = mul_raw(grad_output, output).sum(&[self.dim], true);
        let grad = mul_raw(output, &sub_raw(grad_output, &sum_gs));
        smallvec::smallvec![grad]
    }
}
```

#### Conv1d

```rust
struct Conv1dBackward {
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl BackwardOp for Conv1dBackward {
    fn backward(
        &self,
        grad_output: &Tensor,
        saved: &[Tensor],
    ) -> SmallVec<[Tensor; 2]> {
        let input = &saved[0];  // [batch, in_ch, length]
        let weight = &saved[1]; // [out_ch, in_ch, kernel]

        // grad_input: transposed convolution (deconvolution) of grad_output with weight
        let grad_input = conv1d_backward_input(
            grad_output, weight, input,
            self.stride, self.padding, self.dilation,
        );

        // grad_weight: cross-correlation of input with grad_output
        let grad_weight = conv1d_backward_weight(
            input, grad_output, weight,
            self.stride, self.padding, self.dilation,
        );

        smallvec::smallvec![grad_input, grad_weight]
    }
}
```

---

## 4. Neural Network Layers

### Layer Trait

All layers take `&mut self` (not `&self`) so that stateful layers like `BatchNorm1d` and
`Dropout` can legally mutate their internal state during forward passes.

The optional `tape` parameter controls gradient tracking: pass `Some(&mut tape)` during
training, or `None` during inference.

```rust
pub trait Layer: Send + Sync {
    /// Forward pass. Takes &mut self so stateful layers (BatchNorm, Dropout)
    /// can update internal state.
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor;

    /// Get references to trainable parameters (for optimizer).
    fn parameters(&self) -> Vec<&Tensor>;

    /// Get mutable references to trainable parameters (for optimizer step).
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    /// Switch to training mode.
    fn train(&mut self);

    /// Switch to evaluation mode.
    fn eval(&mut self);

    /// Count parameters: returns (total, frozen).
    fn parameter_count(&self) -> (usize, usize) {
        let params = self.parameters();
        let total: usize = params.iter()
            .map(|p| p.shape.iter().product::<usize>())
            .sum();
        (total, 0) // default: nothing frozen
    }
}

/// Print a structured model summary.
pub fn model_summary(model: &dyn Layer) {
    let (total, frozen) = model.parameter_count();
    let trainable = total - frozen;
    println!("┌─────────────────────────────────────┐");
    println!("│         Model Summary               │");
    println!("├─────────────────────────────────────┤");
    println!("│ Total parameters:     {:>12}  │", total);
    println!("│ Trainable parameters: {:>12}  │", trainable);
    println!("│ Frozen parameters:    {:>12}  │", frozen);
    println!("└─────────────────────────────────────┘");
}
```

### Sequential Container

`Sequential` is a generic container that chains layers. It was referenced but never
defined in the original design.

```rust
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }
}

impl Layer for Sequential {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        let mut x = input.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x, tape.as_deref_mut());
        }
        x
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers.iter_mut().flat_map(|l| l.parameters_mut()).collect()
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn parameter_count(&self) -> (usize, usize) {
        self.layers.iter().fold((0, 0), |(t, f), l| {
            let (lt, lf) = l.parameter_count();
            (t + lt, f + lf)
        })
    }
}
```

### Linear Layer

He initialization is corrected to `(2.0 / in_features as f32).sqrt()`, not
`2.0 / (in_features as f32).sqrt()`. Xavier (`1.0 / (in_features as f32).sqrt()`)
is the default; He is used when followed by ReLU.

```rust
pub enum InitMethod {
    Xavier,
    He,
}

pub struct Linear {
    weight: Tensor,  // [out_features, in_features]
    bias: Option<Tensor>,  // [out_features]
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self::with_init(in_features, out_features, bias, InitMethod::Xavier)
    }

    pub fn with_init(
        in_features: usize,
        out_features: usize,
        bias: bool,
        init: InitMethod,
    ) -> Self {
        let scale = match init {
            // Xavier: sqrt(1 / in_features)
            InitMethod::Xavier => (1.0 / in_features as f32).sqrt(),
            // He: sqrt(2 / in_features) — correct formula
            InitMethod::He => (2.0 / in_features as f32).sqrt(),
        };

        let weight = Tensor::randn(&[out_features, in_features]) * scale;

        let bias = if bias {
            Some(Tensor::zeros(&[out_features]))
        } else {
            None
        };

        Self { weight, bias, in_features, out_features }
    }
}

impl Layer for Linear {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        // input: [batch, in_features]
        // output: [batch, out_features]
        let output = matmul(input, &self.weight.transpose(0, 1), tape);

        if let Some(bias) = &self.bias {
            add(output, bias, tape) // broadcasting
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(bias) = &self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(bias) = &mut self.bias {
            params.push(bias);
        }
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}
```

### LSTM Layer

```rust
pub struct LSTM {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,

    // Parameters for each layer
    weights_ih: Vec<Tensor>,  // Input to hidden [4*hidden, input]
    weights_hh: Vec<Tensor>,  // Hidden to hidden [4*hidden, hidden]
    bias_ih: Vec<Tensor>,     // Input bias [4*hidden]
    bias_hh: Vec<Tensor>,     // Hidden bias [4*hidden]

    dropout: f32,
    bidirectional: bool,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut weights_ih = Vec::new();
        let mut weights_hh = Vec::new();
        let mut bias_ih = Vec::new();
        let mut bias_hh = Vec::new();

        for layer in 0..num_layers {
            let input_dim = if layer == 0 { input_size } else { hidden_size };

            weights_ih.push(Tensor::randn(&[4 * hidden_size, input_dim]));
            weights_hh.push(Tensor::randn(&[4 * hidden_size, hidden_size]));
            bias_ih.push(Tensor::zeros(&[4 * hidden_size]));
            bias_hh.push(Tensor::zeros(&[4 * hidden_size]));
        }

        Self {
            input_size,
            hidden_size,
            num_layers,
            weights_ih,
            weights_hh,
            bias_ih,
            bias_hh,
            dropout: 0.0,
            bidirectional: false,
        }
    }

    pub fn forward(
        &mut self,
        input: &Tensor,
        state: Option<(Tensor, Tensor)>,
        tape: Option<&mut GradientTape>,
    ) -> (Tensor, (Tensor, Tensor)) {
        // input: [seq_len, batch, input_size]
        let seq_len = input.shape()[0];
        let batch_size = input.shape()[1];

        let (mut h, mut c) = state.unwrap_or_else(|| {
            (
                Tensor::zeros(&[self.num_layers, batch_size, self.hidden_size]),
                Tensor::zeros(&[self.num_layers, batch_size, self.hidden_size]),
            )
        });

        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let x = input.slice(&[t..t+1, .., ..]).squeeze(0);

            for layer in 0..self.num_layers {
                let (h_new, c_new) = self.lstm_cell(
                    &x,
                    &h.slice(&[layer..layer+1, .., ..]).squeeze(0),
                    &c.slice(&[layer..layer+1, .., ..]).squeeze(0),
                    layer,
                    tape,
                );

                h = h.index_copy(layer, &h_new);
                c = c.index_copy(layer, &c_new);
            }

            outputs.push(h.slice(&[-1.., .., ..]));
        }

        let output = Tensor::stack(&outputs, 0);
        (output, (h, c))
    }

    fn lstm_cell(
        &self,
        x: &Tensor,
        h: &Tensor,
        c: &Tensor,
        layer: usize,
        tape: Option<&mut GradientTape>,
    ) -> (Tensor, Tensor) {
        // Gates: input, forget, cell, output
        let gates = add(
            &add(
                &matmul(x, &self.weights_ih[layer].t(), tape),
                &self.bias_ih[layer],
                tape,
            ),
            &add(
                &matmul(h, &self.weights_hh[layer].t(), tape),
                &self.bias_hh[layer],
                tape,
            ),
            tape,
        );

        let chunk_size = self.hidden_size;
        let i = gates.slice(&[.., 0..chunk_size]).sigmoid(tape);
        let f = gates.slice(&[.., chunk_size..2*chunk_size]).sigmoid(tape);
        let g = gates.slice(&[.., 2*chunk_size..3*chunk_size]).tanh(tape);
        let o = gates.slice(&[.., 3*chunk_size..4*chunk_size]).sigmoid(tape);

        let c_new = add(&mul(&f, c, tape), &mul(&i, &g, tape), tape);
        let h_new = mul(&o, &c_new.tanh(tape), tape);

        (h_new, c_new)
    }
}
```

### Conv1d Layer

```rust
pub struct Conv1d {
    weight: Tensor,  // [out_channels, in_channels, kernel_size]
    bias: Option<Tensor>,  // [out_channels]
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        let weight = Tensor::randn(&[out_channels, in_channels, kernel_size]) *
                     (2.0 / (in_channels * kernel_size) as f32).sqrt();
        let bias = Some(Tensor::zeros(&[out_channels]));

        Self {
            weight, bias, in_channels, out_channels,
            kernel_size, stride, padding, dilation,
        }
    }
}

impl Layer for Conv1d {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        // input: [batch, in_channels, length]
        // output: [batch, out_channels, length_out]
        conv1d_op(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            tape,
        )
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(bias) = &self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(bias) = &mut self.bias {
            params.push(bias);
        }
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}
```

### Activation Functions

```rust
pub struct ReLU;
impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor, tape: Option<&mut GradientTape>) -> Tensor {
        relu_op(input, tape)
    }
    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

pub struct Tanh;
impl Layer for Tanh {
    fn forward(&mut self, input: &Tensor, tape: Option<&mut GradientTape>) -> Tensor {
        tanh_op(input, tape)
    }
    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

pub struct Sigmoid;
impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor, tape: Option<&mut GradientTape>) -> Tensor {
        sigmoid_op(input, tape)
    }
    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

pub struct GELU;
impl Layer for GELU {
    fn forward(&mut self, input: &Tensor, tape: Option<&mut GradientTape>) -> Tensor {
        // GELU(x) = x * Φ(x)
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        gelu_op(input, tape)
    }
    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

pub struct SiLU;
impl Layer for SiLU {
    fn forward(&mut self, input: &Tensor, tape: Option<&mut GradientTape>) -> Tensor {
        // SiLU(x) = x * sigmoid(x)
        let s = sigmoid_op(input, tape);
        mul(input, &s, tape)
    }
    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}
```

### Normalization Layers

```rust
pub struct LayerNorm {
    normalized_shape: Shape,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Self {
        let shape: Shape = normalized_shape.into();
        let weight = Tensor::ones(&shape);
        let bias = Tensor::zeros(&shape);
        Self { normalized_shape: shape, weight, bias, eps }
    }
}

impl Layer for LayerNorm {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        let mean = input.mean(&[-1], true);
        let var = input.var(&[-1], true);
        let normalized = div(
            &sub(input, &mean, tape),
            &add_scalar(&var.sqrt(), self.eps, tape),
            tape,
        );
        add(&mul(&normalized, &self.weight, tape), &self.bias, tape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}

pub struct BatchNorm1d {
    num_features: usize,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    momentum: f32,
    training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            weight: Tensor::ones(&[num_features]),
            bias: Tensor::zeros(&[num_features]),
            eps: 1e-5,
            momentum: 0.1,
            training: true,
        }
    }
}

impl Layer for BatchNorm1d {
    /// forward takes &mut self — this is essential because training mode
    /// updates running_mean and running_var.
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        if self.training {
            let mean = input.mean(&[0], false);
            let var = input.var(&[0], false);

            // Update running statistics (&mut self allows this)
            self.running_mean = add_raw(
                &mul_scalar_raw(&self.running_mean, 1.0 - self.momentum),
                &mul_scalar_raw(&mean, self.momentum),
            );
            self.running_var = add_raw(
                &mul_scalar_raw(&self.running_var, 1.0 - self.momentum),
                &mul_scalar_raw(&var, self.momentum),
            );

            let normalized = div(
                &sub(input, &mean, tape),
                &add_scalar(&var.sqrt(), self.eps, tape),
                tape,
            );
            add(&mul(&normalized, &self.weight, tape), &self.bias, tape)
        } else {
            let normalized = div(
                &sub(input, &self.running_mean, tape),
                &add_scalar(&self.running_var.sqrt(), self.eps, tape),
                tape,
            );
            add(&mul(&normalized, &self.weight, tape), &self.bias, tape)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}
```

### Dropout

```rust
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self { p, training: true }
    }
}

impl Layer for Dropout {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        if self.training && self.p > 0.0 {
            let mask = Tensor::rand_like(input).greater_than(self.p);
            div(&mul(input, &mask, tape), &Tensor::scalar(1.0 - self.p), tape)
        } else {
            input.clone()
        }
    }

    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![] }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}
```

---

## 5. Optimizers

### Optimizer Trait

Optimizers read gradients from the `GradientTape` rather than from `param.grad`.

```rust
pub trait Optimizer {
    /// Perform a single optimization step.
    ///
    /// `params`: mutable references to all trainable parameter tensors.
    /// `tape`: the gradient tape containing computed gradients.
    fn step(&mut self, params: &mut [&mut Tensor], tape: &GradientTape);

    /// Get current learning rate.
    fn lr(&self) -> f32;

    /// Set learning rate.
    fn set_lr(&mut self, lr: f32);
}
```

### Gradient Clipping

```rust
pub enum GradClip {
    /// Clip by global L2 norm.
    Norm(f32),
    /// Clip each gradient element to [-value, value].
    Value(f32),
}

/// Clip gradients by global L2 norm. Returns the original norm.
pub fn clip_grad_norm(tape: &mut GradientTape, param_ids: &[TensorId], max_norm: f32) -> f32 {
    // Compute global norm
    let total_norm_sq: f32 = param_ids.iter()
        .filter_map(|id| tape.grad(*id))
        .map(|g| g.pow(2.0).sum(&[], false).item())
        .sum();
    let total_norm = total_norm_sq.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for id in param_ids {
            if let Some(grad) = tape.grads.get_mut(id) {
                *grad = mul_scalar_raw(grad, scale);
            }
        }
    }

    total_norm
}

/// Clip each gradient element to [-value, value].
pub fn clip_grad_value(tape: &mut GradientTape, param_ids: &[TensorId], clip_value: f32) {
    for id in param_ids {
        if let Some(grad) = tape.grads.get_mut(id) {
            *grad = grad.clamp(-clip_value, clip_value);
        }
    }
}
```

### SGD

```rust
pub struct SGD {
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: HashMap<TensorId, Tensor>,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            momentum,
            dampening: 0.0,
            weight_decay,
            nesterov: false,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut Tensor], tape: &GradientTape) {
        for param in params.iter_mut() {
            let grad = match tape.grad(param.id) {
                Some(g) => g.clone(),
                None => continue,
            };

            let mut d_p = grad;

            // Weight decay
            if self.weight_decay != 0.0 {
                d_p = add_raw(&d_p, &mul_scalar_raw(param, self.weight_decay));
            }

            // Momentum
            if self.momentum != 0.0 {
                let v = self.velocity.entry(param.id)
                    .or_insert_with(|| Tensor::zeros_like(param));
                *v = add_raw(
                    &mul_scalar_raw(v, self.momentum),
                    &mul_scalar_raw(&d_p, 1.0 - self.dampening),
                );

                if self.nesterov {
                    d_p = add_raw(&d_p, &mul_scalar_raw(v, self.momentum));
                } else {
                    d_p = v.clone();
                }
            }

            // Update parameter
            **param = sub_raw(param, &mul_scalar_raw(&d_p, self.lr));
        }
    }

    fn lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
}
```

### Adam

```rust
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step_count: usize,
    m: HashMap<TensorId, Tensor>,  // First moment
    v: HashMap<TensorId, Tensor>,  // Second moment
}

impl Adam {
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            weight_decay,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut Tensor], tape: &GradientTape) {
        self.step_count += 1;

        for param in params.iter_mut() {
            let grad = match tape.grad(param.id) {
                Some(g) => g.clone(),
                None => continue,
            };

            let mut g = grad;

            // L2 weight decay (Adam-style, applied to gradient)
            if self.weight_decay != 0.0 {
                g = add_raw(&g, &mul_scalar_raw(param, self.weight_decay));
            }

            // First moment
            let m = self.m.entry(param.id)
                .or_insert_with(|| Tensor::zeros_like(param));
            *m = add_raw(
                &mul_scalar_raw(m, self.beta1),
                &mul_scalar_raw(&g, 1.0 - self.beta1),
            );

            // Second moment
            let v = self.v.entry(param.id)
                .or_insert_with(|| Tensor::zeros_like(param));
            *v = add_raw(
                &mul_scalar_raw(v, self.beta2),
                &mul_scalar_raw(&mul_raw(&g, &g), 1.0 - self.beta2),
            );

            // Bias correction
            let m_hat = mul_scalar_raw(m, 1.0 / (1.0 - self.beta1.powi(self.step_count as i32)));
            let v_hat = mul_scalar_raw(v, 1.0 / (1.0 - self.beta2.powi(self.step_count as i32)));

            // Update parameters
            let update = div_raw(&m_hat, &add_scalar_raw(&v_hat.sqrt(), self.eps));
            **param = sub_raw(param, &mul_scalar_raw(&update, self.lr));
        }
    }

    fn lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
}
```

### AdamW

```rust
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step_count: usize,
    m: HashMap<TensorId, Tensor>,
    v: HashMap<TensorId, Tensor>,
}

impl AdamW {
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            weight_decay,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [&mut Tensor], tape: &GradientTape) {
        self.step_count += 1;

        for param in params.iter_mut() {
            let grad = match tape.grad(param.id) {
                Some(g) => g.clone(),
                None => continue,
            };

            // First moment
            let m = self.m.entry(param.id)
                .or_insert_with(|| Tensor::zeros_like(param));
            *m = add_raw(
                &mul_scalar_raw(m, self.beta1),
                &mul_scalar_raw(&grad, 1.0 - self.beta1),
            );

            // Second moment
            let v = self.v.entry(param.id)
                .or_insert_with(|| Tensor::zeros_like(param));
            *v = add_raw(
                &mul_scalar_raw(v, self.beta2),
                &mul_scalar_raw(&mul_raw(&grad, &grad), 1.0 - self.beta2),
            );

            // Bias correction
            let m_hat = mul_scalar_raw(m, 1.0 / (1.0 - self.beta1.powi(self.step_count as i32)));
            let v_hat = mul_scalar_raw(v, 1.0 / (1.0 - self.beta2.powi(self.step_count as i32)));

            // Decoupled weight decay: subtract lr * weight_decay * param FIRST
            **param = mul_scalar_raw(param, 1.0 - self.lr * self.weight_decay);

            // Then apply Adam update
            let update = div_raw(&m_hat, &add_scalar_raw(&v_hat.sqrt(), self.eps));
            **param = sub_raw(param, &mul_scalar_raw(&update, self.lr));
        }
    }

    fn lr(&self) -> f32 { self.lr }
    fn set_lr(&mut self, lr: f32) { self.lr = lr; }
}
```

### Learning Rate Schedulers

```rust
pub trait LRScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer);
    fn get_lr(&self) -> f32;
}

pub struct StepLR {
    step_size: usize,
    gamma: f32,
    current_step: usize,
    base_lr: f32,
}

impl StepLR {
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self { step_size, gamma, current_step: 0, base_lr }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        if self.current_step % self.step_size == 0 {
            let new_lr = optimizer.lr() * self.gamma;
            optimizer.set_lr(new_lr);
        }
    }

    fn get_lr(&self) -> f32 {
        self.base_lr * self.gamma.powi((self.current_step / self.step_size) as i32)
    }
}

pub struct CosineAnnealingLR {
    t_max: usize,
    eta_min: f32,
    current_step: usize,
    base_lr: f32,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, t_max: usize, eta_min: f32) -> Self {
        Self { t_max, eta_min, current_step: 0, base_lr }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let new_lr = self.get_lr();
        optimizer.set_lr(new_lr);
    }

    fn get_lr(&self) -> f32 {
        let progress = self.current_step as f32 / self.t_max as f32;
        let cosine = (1.0 + (progress * std::f32::consts::PI).cos()) / 2.0;
        self.eta_min + (self.base_lr - self.eta_min) * cosine
    }
}

/// Linear warmup followed by cosine decay.
pub struct WarmupCosineScheduler {
    warmup_steps: usize,
    total_steps: usize,
    base_lr: f32,
    eta_min: f32,
    current_step: usize,
}

impl WarmupCosineScheduler {
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize, eta_min: f32) -> Self {
        Self {
            warmup_steps,
            total_steps,
            base_lr,
            eta_min,
            current_step: 0,
        }
    }
}

impl LRScheduler for WarmupCosineScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        optimizer.set_lr(self.get_lr());
    }

    fn get_lr(&self) -> f32 {
        if self.current_step <= self.warmup_steps {
            // Linear warmup: 0 -> base_lr
            self.base_lr * (self.current_step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine decay: base_lr -> eta_min
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_progress =
                (self.current_step - self.warmup_steps) as f32 / decay_steps as f32;
            let cosine = (1.0 + (decay_progress * std::f32::consts::PI).cos()) / 2.0;
            self.eta_min + (self.base_lr - self.eta_min) * cosine
        }
    }
}
```

---

## 6. Loss Functions

Loss functions operate on tensors and record to the tape automatically through the
underlying tensor ops. No signature changes are needed versus pure-inference tensor math —
the tape propagation happens inside `sub`, `mul`, `pow`, etc.

```rust
pub trait Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor, tape: Option<&mut GradientTape>) -> Tensor;
}

pub struct MSELoss;

impl Loss for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor, tape: Option<&mut GradientTape>) -> Tensor {
        let diff = sub(predictions, targets, tape);
        pow(&diff, 2.0, tape).mean(&[], false)
    }
}

pub struct MAELoss;

impl Loss for MAELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor, tape: Option<&mut GradientTape>) -> Tensor {
        sub(predictions, targets, tape).abs(tape).mean(&[], false)
    }
}

pub struct HuberLoss {
    delta: f32,
}

impl HuberLoss {
    pub fn new(delta: f32) -> Self {
        Self { delta }
    }
}

impl Loss for HuberLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor, tape: Option<&mut GradientTape>) -> Tensor {
        let diff = sub(predictions, targets, tape).abs(tape);
        let quadratic = mul_scalar(&pow(&diff, 2.0, tape), 0.5, tape);
        let linear = sub_scalar(
            &mul_scalar(&diff, self.delta, tape),
            self.delta * self.delta * 0.5,
            tape,
        );

        // Use quadratic where |diff| <= delta, linear otherwise
        let mask = diff.less_equal(self.delta);
        add(
            &mul(&mask, &quadratic, tape),
            &mul(&sub_scalar(&Tensor::ones_like(&mask), 1.0, tape), &linear, tape),
            tape,
        ).mean(&[], false)
    }
}

pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn forward(&self, logits: &Tensor, targets: &Tensor, tape: Option<&mut GradientTape>) -> Tensor {
        // logits: [batch, num_classes]
        // targets: [batch] (class indices)

        // Numerically stable log-softmax
        let max_logits = logits.max(&[-1], true);
        let shifted = sub(logits, &max_logits, tape);
        let exp_logits = shifted.exp(tape);
        let log_sum_exp = add(&exp_logits.sum(&[-1], true).log(tape), &max_logits, tape);

        // Gather target logits
        let target_logits = logits.gather(-1, targets);

        // Cross entropy
        sub(&log_sum_exp, &target_logits, tape).mean(&[], false)
    }
}

pub struct QuantileLoss {
    quantile: f32,
}

impl QuantileLoss {
    pub fn new(quantile: f32) -> Self {
        assert!(quantile > 0.0 && quantile < 1.0);
        Self { quantile }
    }
}

impl Loss for QuantileLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor, tape: Option<&mut GradientTape>) -> Tensor {
        let errors = sub(targets, predictions, tape);
        let positive_errors = errors.maximum(&Tensor::zeros_like(&errors));
        let negative_errors = neg(&errors, tape).maximum(&Tensor::zeros_like(&errors));

        add(
            &mul_scalar(&positive_errors, self.quantile, tape),
            &mul_scalar(&negative_errors, 1.0 - self.quantile, tape),
            tape,
        ).mean(&[], false)
    }
}
```

---

## 7. Time Series Components

### Data Structures

```rust
pub struct OHLCVCandle {
    pub timestamp: i64,
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
}

/// Configurable target column(s) for multi-feature target support.
pub enum TargetColumn {
    /// Single column by name.
    Single(String),
    /// Multiple columns.
    Multi(Vec<String>),
    /// Close price (default).
    Close,
}

pub struct TimeSeriesDataset {
    data: Vec<OHLCVCandle>,
    window_size: usize,
    prediction_horizon: usize,

    // Feature engineering
    features: Vec<String>,

    // Target configuration
    target_columns: TargetColumn,

    // Normalization
    scaler: Option<Scaler>,
}

impl TimeSeriesDataset {
    pub fn new(data: Vec<OHLCVCandle>, window_size: usize, horizon: usize) -> Self {
        Self {
            data,
            window_size,
            prediction_horizon: horizon,
            features: vec!["open", "high", "low", "close", "volume"]
                .iter().map(|s| s.to_string()).collect(),
            target_columns: TargetColumn::Close,
            scaler: None,
        }
    }

    pub fn with_targets(mut self, targets: TargetColumn) -> Self {
        self.target_columns = targets;
        self
    }

    pub fn len(&self) -> usize {
        self.data.len().saturating_sub(self.window_size + self.prediction_horizon - 1)
    }

    pub fn get(&self, idx: usize) -> (Tensor, Tensor) {
        let input_start = idx;
        let input_end = idx + self.window_size;

        let input_data: Vec<f32> = self.data[input_start..input_end]
            .iter()
            .flat_map(|candle| vec![
                candle.open, candle.high, candle.low, candle.close, candle.volume,
            ])
            .collect();

        let input = Tensor::from_vec(input_data, &[self.window_size, 5]);

        // Target based on configured target column(s)
        let target_idx = input_end + self.prediction_horizon - 1;
        let target_candle = &self.data[target_idx];
        let target = match &self.target_columns {
            TargetColumn::Close => {
                Tensor::from_vec(vec![target_candle.close], &[1])
            }
            TargetColumn::Single(col) => {
                let val = match col.as_str() {
                    "open" => target_candle.open,
                    "high" => target_candle.high,
                    "low" => target_candle.low,
                    "close" => target_candle.close,
                    "volume" => target_candle.volume,
                    _ => target_candle.close,
                };
                Tensor::from_vec(vec![val], &[1])
            }
            TargetColumn::Multi(cols) => {
                let vals: Vec<f32> = cols.iter().map(|col| {
                    match col.as_str() {
                        "open" => target_candle.open,
                        "high" => target_candle.high,
                        "low" => target_candle.low,
                        "close" => target_candle.close,
                        "volume" => target_candle.volume,
                        _ => 0.0,
                    }
                }).collect();
                let n = vals.len();
                Tensor::from_vec(vals, &[n])
            }
        };

        (input, target)
    }
}
```

### Feature Engineering

```rust
pub struct FeatureEngineer {
    features: Vec<Box<dyn Feature>>,
}

pub trait Feature {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32>;
    fn name(&self) -> &str;
}

pub struct Returns;
impl Feature for Returns {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        data.windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect()
    }
    fn name(&self) -> &str { "returns" }
}

pub struct MovingAverage {
    window: usize,
}

impl Feature for MovingAverage {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        data.windows(self.window)
            .map(|w| w.iter().map(|c| c.close).sum::<f32>() / self.window as f32)
            .collect()
    }
    fn name(&self) -> &str { "sma" }
}

pub struct Volatility {
    window: usize,
}

impl Feature for Volatility {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        let returns: Vec<f32> = data.windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        returns.windows(self.window)
            .map(|w| {
                let mean = w.iter().sum::<f32>() / w.len() as f32;
                let variance = w.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f32>() / w.len() as f32;
                variance.sqrt()
            })
            .collect()
    }
    fn name(&self) -> &str { "volatility" }
}
```

### RSI — Wilder's Exponential Moving Average

The original used simple moving average for RSI, which is incorrect. Wilder's RSI
uses an exponential smoothing factor of `1/period`:

```rust
pub struct RSI {
    period: usize,
}

impl Feature for RSI {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        if data.len() < self.period + 1 {
            return vec![];
        }

        let mut gains = Vec::with_capacity(data.len() - 1);
        let mut losses = Vec::with_capacity(data.len() - 1);

        for window in data.windows(2) {
            let change = window[1].close - window[0].close;
            gains.push(change.max(0.0));
            losses.push((-change).max(0.0));
        }

        // Wilder's smoothing: initial seed is SMA of first `period` values,
        // then exponential: avg = prev_avg * (period-1)/period + current / period
        let alpha = 1.0 / self.period as f32;

        // Seed with SMA of first `period` values
        let mut avg_gain: f32 = gains[..self.period].iter().sum::<f32>() / self.period as f32;
        let mut avg_loss: f32 = losses[..self.period].iter().sum::<f32>() / self.period as f32;

        let mut rsi_values = Vec::with_capacity(gains.len() - self.period + 1);

        // First RSI value from the SMA seed
        let rs = if avg_loss == 0.0 { f32::MAX } else { avg_gain / avg_loss };
        rsi_values.push(100.0 - (100.0 / (1.0 + rs)));

        // Subsequent values use Wilder's EMA
        for i in self.period..gains.len() {
            avg_gain = avg_gain * (1.0 - alpha) + gains[i] * alpha;
            avg_loss = avg_loss * (1.0 - alpha) + losses[i] * alpha;

            let rs = if avg_loss == 0.0 { f32::MAX } else { avg_gain / avg_loss };
            rsi_values.push(100.0 - (100.0 / (1.0 + rs)));
        }

        rsi_values
    }
    fn name(&self) -> &str { "rsi" }
}
```

### Normalization (Scaler)

```rust
pub enum ScalerType {
    MinMax,
    StandardScaler,
    RobustScaler,
}

pub struct Scaler {
    scaler_type: ScalerType,
    params: Vec<(f32, f32)>,  // Per-feature parameters
}

impl Scaler {
    pub fn fit(data: &Tensor, scaler_type: ScalerType) -> Self {
        let num_features = data.shape()[1];
        let mut params = Vec::new();

        for feat_idx in 0..num_features {
            let feature = data.slice(&[.., feat_idx..feat_idx+1]);

            let (p1, p2) = match scaler_type {
                ScalerType::MinMax => {
                    let min = feature.min(&[], false).item();
                    let max = feature.max(&[], false).item();
                    (min, max - min)
                }
                ScalerType::StandardScaler => {
                    let mean = feature.mean(&[], false).item();
                    let std = feature.std(&[], false).item();
                    (mean, std)
                }
                ScalerType::RobustScaler => {
                    let median = feature.median();
                    let q75 = feature.quantile(0.75);
                    let q25 = feature.quantile(0.25);
                    (median, q75 - q25)
                }
            };

            params.push((p1, p2));
        }

        Self { scaler_type, params }
    }

    pub fn transform(&self, data: &Tensor) -> Tensor {
        let mut result = data.clone();
        for (feat_idx, (p1, p2)) in self.params.iter().enumerate() {
            let feature = data.slice(&[.., feat_idx..feat_idx+1]);
            let transformed = (feature - *p1) / *p2;
            result = result.index_copy(feat_idx, &transformed);
        }
        result
    }

    pub fn inverse_transform(&self, data: &Tensor) -> Tensor {
        let mut result = data.clone();
        for (feat_idx, (p1, p2)) in self.params.iter().enumerate() {
            let feature = data.slice(&[.., feat_idx..feat_idx+1]);
            let original = feature * *p2 + *p1;
            result = result.index_copy(feat_idx, &original);
        }
        result
    }
}
```

### DataLoader — Implements Iterator

```rust
pub struct DataLoader {
    dataset: TimeSeriesDataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_idx: usize,
}

impl DataLoader {
    pub fn new(dataset: TimeSeriesDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_idx: 0,
        }
    }

    /// Reset for a new epoch. Re-shuffles if shuffle is enabled.
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }
}

/// DataLoader implements Iterator, yielding (input_batch, target_batch) tuples.
impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];

        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for &idx in batch_indices {
            let (input, target) = self.dataset.get(idx);
            inputs.push(input);
            targets.push(target);
        }

        self.current_idx = end_idx;

        Some((
            Tensor::stack(&inputs, 0),
            Tensor::stack(&targets, 0),
        ))
    }
}
```

---

## 8. Model Architectures

### TCN (Temporal Convolutional Network)

Causal padding is `(kernel_size - 1) * dilation` applied entirely on the **left** side.
After convolution, the output is trimmed to match the input length. This ensures no
information from future time steps leaks into the output.

```rust
pub struct TCNBlock {
    conv1: Conv1d,
    conv2: Conv1d,
    downsample: Option<Conv1d>,
    relu: ReLU,
    dropout: Dropout,
    /// Amount to trim from the right after convolution.
    trim1: usize,
    trim2: usize,
}

impl TCNBlock {
    pub fn new(
        n_inputs: usize,
        n_outputs: usize,
        kernel_size: usize,
        dilation: usize,
        dropout: f32,
    ) -> Self {
        // Causal padding: full left pad = (kernel_size - 1) * dilation
        let causal_pad = (kernel_size - 1) * dilation;

        let mut conv1 = Conv1d::new(n_inputs, n_outputs, kernel_size, 1, causal_pad, dilation);
        let mut conv2 = Conv1d::new(n_outputs, n_outputs, kernel_size, 1, causal_pad, dilation);

        let downsample = if n_inputs != n_outputs {
            Some(Conv1d::new(n_inputs, n_outputs, 1, 1, 0, 1))
        } else {
            None
        };

        Self {
            conv1,
            conv2,
            downsample,
            relu: ReLU,
            dropout: Dropout::new(dropout),
            trim1: causal_pad,
            trim2: causal_pad,
        }
    }
}

impl Layer for TCNBlock {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        // input: [batch, channels, length]
        let length = input.shape()[2];

        let mut out = self.conv1.forward(input, tape);
        // Trim right side to enforce causality: output length = input length
        out = out.slice(&[.., .., ..length]);
        out = self.relu.forward(&out, tape);
        out = self.dropout.forward(&out, tape);

        out = self.conv2.forward(&out, tape);
        out = out.slice(&[.., .., ..length]);
        out = self.relu.forward(&out, tape);
        out = self.dropout.forward(&out, tape);

        let res = if let Some(downsample) = &mut self.downsample {
            downsample.forward(input, tape)
        } else {
            input.clone()
        };

        self.relu.forward(&add(&out, &res, tape), tape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        if let Some(downsample) = &self.downsample {
            params.extend(downsample.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        if let Some(downsample) = &mut self.downsample {
            params.extend(downsample.parameters_mut());
        }
        params
    }

    fn train(&mut self) {
        self.conv1.train();
        self.conv2.train();
        self.dropout.train();
        if let Some(d) = &mut self.downsample { d.train(); }
    }

    fn eval(&mut self) {
        self.conv1.eval();
        self.conv2.eval();
        self.dropout.eval();
        if let Some(d) = &mut self.downsample { d.eval(); }
    }
}

pub struct TCN {
    blocks: Vec<TCNBlock>,
    fc: Linear,
}

impl TCN {
    pub fn new(
        input_size: usize,
        num_channels: Vec<usize>,
        kernel_size: usize,
        dropout: f32,
        output_size: usize,
    ) -> Self {
        let mut blocks = Vec::new();
        let mut in_channels = input_size;

        for (i, &out_channels) in num_channels.iter().enumerate() {
            let dilation = 2_usize.pow(i as u32);
            blocks.push(TCNBlock::new(
                in_channels,
                out_channels,
                kernel_size,
                dilation,
                dropout,
            ));
            in_channels = out_channels;
        }

        let fc = Linear::new(*num_channels.last().unwrap(), output_size, true);

        Self { blocks, fc }
    }
}

impl Layer for TCN {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        let mut out = input.clone();
        for block in &mut self.blocks {
            out = block.forward(&out, tape);
        }

        // Global average pooling over time dimension
        out = out.mean(&[2], false);

        self.fc.forward(&out, tape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.fc.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
        }
        params.extend(self.fc.parameters_mut());
        params
    }

    fn train(&mut self) {
        for block in &mut self.blocks { block.train(); }
        self.fc.train();
    }

    fn eval(&mut self) {
        for block in &mut self.blocks { block.eval(); }
        self.fc.eval();
    }
}
```

### Transformer for Time Series

#### Attention Masking

```rust
/// Create a causal (autoregressive) mask: upper-triangle is -inf.
/// Shape: [seq_len, seq_len].
pub fn causal_mask(seq_len: usize) -> Tensor {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, &[seq_len, seq_len])
}

/// Create a padding mask from sequence lengths.
/// Returns shape [batch, 1, 1, max_len] where padded positions are -inf.
pub fn padding_mask(lengths: &[usize], max_len: usize) -> Tensor {
    let batch = lengths.len();
    let mut data = vec![0.0f32; batch * max_len];
    for (b, &len) in lengths.iter().enumerate() {
        for j in len..max_len {
            data[b * max_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, &[batch, 1, 1, max_len])
}

/// Combine causal and padding masks (element-wise minimum, since both use -inf).
pub fn combined_mask(causal: &Tensor, padding: &Tensor) -> Tensor {
    add_raw(causal, padding) // -inf + 0 = -inf, 0 + 0 = 0
}
```

#### Multi-Head Attention

```rust
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(d_model % num_heads, 0);
        let head_dim = d_model / num_heads;

        Self {
            num_heads,
            head_dim,
            q_proj: Linear::new(d_model, d_model, true),
            k_proj: Linear::new(d_model, d_model, true),
            v_proj: Linear::new(d_model, d_model, true),
            out_proj: Linear::new(d_model, d_model, true),
        }
    }

    pub fn forward_attn(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];

        // Project and reshape to [batch, heads, seq_len, head_dim]
        let q = self.q_proj.forward(query, tape)
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        let k = self.k_proj.forward(key, tape)
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        let v = self.v_proj.forward(value, tape)
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        // Attention scores: [batch, heads, seq_len, seq_len]
        let scale = (self.head_dim as f32).sqrt();
        let scores = div_scalar(&matmul(&q, &k.transpose(-2, -1), tape), scale, tape);

        // Apply mask if provided (causal, padding, or combined)
        let scores = if let Some(mask) = mask {
            add(&scores, mask, tape)
        } else {
            scores
        };

        // Softmax over last dim
        let attn_weights = softmax(&scores, -1, tape);

        // Apply attention to values
        let out = matmul(&attn_weights, &v, tape);

        // Reshape back: [batch, seq_len, d_model]
        let out = out.transpose(1, 2)
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim]);

        self.out_proj.forward(&out, tape)
    }
}
```

#### Transformer Block

```rust
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    ffn: Sequential,
    dropout: Dropout,
}

impl TransformerBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout: f32) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads),
            norm1: LayerNorm::new(vec![d_model], 1e-5),
            norm2: LayerNorm::new(vec![d_model], 1e-5),
            ffn: Sequential::new(vec![
                Box::new(Linear::new(d_model, d_ff, true)),
                Box::new(GELU),
                Box::new(Linear::new(d_ff, d_model, true)),
            ]),
            dropout: Dropout::new(dropout),
        }
    }

    pub fn forward_with_mask(
        &mut self,
        input: &Tensor,
        mask: Option<&Tensor>,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        // Self-attention with residual
        let attn_out = self.attention.forward_attn(input, input, input, mask, tape);
        let attn_out = self.dropout.forward(&attn_out, tape);
        let x = self.norm1.forward(&add(input, &attn_out, tape), tape);

        // Feed-forward with residual
        let ffn_out = self.ffn.forward(&x, tape);
        let ffn_out = self.dropout.forward(&ffn_out, tape);
        self.norm2.forward(&add(&x, &ffn_out, tape), tape)
    }
}

impl Layer for TransformerBlock {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        self.forward_with_mask(input, None, tape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.q_proj.parameters());
        params.extend(self.attention.k_proj.parameters());
        params.extend(self.attention.v_proj.parameters());
        params.extend(self.attention.out_proj.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.ffn.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.q_proj.parameters_mut());
        params.extend(self.attention.k_proj.parameters_mut());
        params.extend(self.attention.v_proj.parameters_mut());
        params.extend(self.attention.out_proj.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.ffn.parameters_mut());
        params
    }

    fn train(&mut self) { self.dropout.train(); }
    fn eval(&mut self) { self.dropout.eval(); }
}

pub struct TimeSeriesTransformer {
    embedding: Linear,
    positional_encoding: Tensor,
    blocks: Vec<TransformerBlock>,
    head: Linear,
}

impl TimeSeriesTransformer {
    pub fn new(
        input_size: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        d_ff: usize,
        max_seq_len: usize,
        output_size: usize,
        dropout: f32,
    ) -> Self {
        let embedding = Linear::new(input_size, d_model, true);

        // Sinusoidal positional encoding
        let positional_encoding = Self::create_positional_encoding(max_seq_len, d_model);

        let blocks: Vec<_> = (0..num_layers)
            .map(|_| TransformerBlock::new(d_model, num_heads, d_ff, dropout))
            .collect();

        let head = Linear::new(d_model, output_size, true);

        Self { embedding, positional_encoding, blocks, head }
    }

    fn create_positional_encoding(max_len: usize, d_model: usize) -> Tensor {
        let mut pe = vec![0.0; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / d_model as f32);
                pe[pos * d_model + i] = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }

        Tensor::from_vec(pe, &[max_len, d_model])
    }
}

impl Layer for TimeSeriesTransformer {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        let seq_len = input.shape()[1];

        // Embed input
        let mut x = self.embedding.forward(input, tape);

        // Add positional encoding
        let pos_enc = self.positional_encoding.slice(&[0..seq_len, ..]);
        x = add(&x, &pos_enc, tape);

        // Create causal mask for autoregressive attention
        let mask = causal_mask(seq_len);

        // Transformer blocks with mask
        for block in &mut self.blocks {
            x = block.forward_with_mask(&x, Some(&mask), tape);
        }

        // Take last timestep and project
        let last = x.slice(&[.., -1.., ..]).squeeze(1);
        self.head.forward(&last, tape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters_mut());
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
        }
        params.extend(self.head.parameters_mut());
        params
    }

    fn train(&mut self) {
        for block in &mut self.blocks { block.train(); }
    }

    fn eval(&mut self) {
        for block in &mut self.blocks { block.eval(); }
    }
}
```

### N-BEATS

The key fix: `forward_block()` returns **(backcast, forecast)** as a tuple, and
the main forward loop **subtracts the backcast** from the residual signal.

```rust
pub struct NBeatsBlock {
    fc_stack: Vec<Linear>,
    theta_b: Linear,  // Backcast coefficients
    theta_f: Linear,  // Forecast coefficients
    backcast_size: usize,
    forecast_size: usize,
}

impl NBeatsBlock {
    pub fn new(
        input_size: usize,
        theta_size: usize,
        num_layers: usize,
        layer_size: usize,
        backcast_size: usize,
        forecast_size: usize,
    ) -> Self {
        let mut fc_stack = Vec::new();

        fc_stack.push(Linear::with_init(input_size, layer_size, true, InitMethod::He));
        for _ in 1..num_layers {
            fc_stack.push(Linear::with_init(layer_size, layer_size, true, InitMethod::He));
        }

        let theta_b = Linear::new(layer_size, theta_size, true);
        let theta_f = Linear::new(layer_size, theta_size, true);

        Self { fc_stack, theta_b, theta_f, backcast_size, forecast_size }
    }

    /// Returns (backcast, forecast) — both must be used by the caller.
    pub fn forward_block(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> (Tensor, Tensor) {
        let mut x = input.clone();

        // Fully connected stack with ReLU
        for fc in &mut self.fc_stack {
            x = fc.forward(&x, tape);
            x = relu_op(&x, tape);
        }

        // Generate backcast and forecast from theta coefficients
        let backcast = self.theta_b.forward(&x, tape);
        let forecast = self.theta_f.forward(&x, tape);

        (backcast, forecast)
    }
}

pub struct NBeats {
    stacks: Vec<Vec<NBeatsBlock>>,
    forecast_size: usize,
}

impl NBeats {
    pub fn new(
        input_size: usize,
        output_size: usize,
        num_stacks: usize,
        num_blocks_per_stack: usize,
        num_layers: usize,
        layer_size: usize,
    ) -> Self {
        let mut stacks = Vec::new();

        for _ in 0..num_stacks {
            let mut stack = Vec::new();
            for _ in 0..num_blocks_per_stack {
                stack.push(NBeatsBlock::new(
                    input_size,
                    output_size,
                    num_layers,
                    layer_size,
                    input_size,
                    output_size,
                ));
            }
            stacks.push(stack);
        }

        Self { stacks, forecast_size: output_size }
    }
}

impl Layer for NBeats {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        let mut backcast = input.clone();
        let mut forecast = Tensor::zeros(&[input.shape()[0], self.forecast_size]);

        for stack in &mut self.stacks {
            for block in stack {
                let (block_backcast, block_forecast) = block.forward_block(&backcast, tape);

                // Accumulate forecast
                forecast = add(&forecast, &block_forecast, tape);

                // SUBTRACT backcast from residual — this is the N-BEATS doubly-residual design
                backcast = sub(&backcast, &block_backcast, tape);
            }
        }

        forecast
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for stack in &self.stacks {
            for block in stack {
                for fc in &block.fc_stack {
                    params.extend(fc.parameters());
                }
                params.extend(block.theta_b.parameters());
                params.extend(block.theta_f.parameters());
            }
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for stack in &mut self.stacks {
            for block in stack {
                for fc in &mut block.fc_stack {
                    params.extend(fc.parameters_mut());
                }
                params.extend(block.theta_b.parameters_mut());
                params.extend(block.theta_f.parameters_mut());
            }
        }
        params
    }

    fn train(&mut self) {
        for stack in &mut self.stacks {
            for block in stack {
                for fc in &mut block.fc_stack { fc.train(); }
            }
        }
    }

    fn eval(&mut self) {
        for stack in &mut self.stacks {
            for block in stack {
                for fc in &mut block.fc_stack { fc.eval(); }
            }
        }
    }
}
```

---

## 9. Training Infrastructure

### Trainer

`predict` takes `&mut self` so it can call `model.eval()`.

```rust
use std::path::PathBuf;

pub struct Trainer {
    model: Box<dyn Layer>,
    optimizer: Box<dyn Optimizer>,
    loss_fn: Box<dyn Loss>,
    scheduler: Option<Box<dyn LRScheduler>>,
    grad_clip: Option<GradClip>,

    // Metrics
    train_losses: Vec<f32>,
    val_losses: Vec<f32>,

    // Early stopping
    patience: Option<usize>,
    best_val_loss: f32,
    patience_counter: usize,

    // Checkpointing
    checkpoint_dir: Option<PathBuf>,
}

impl Trainer {
    pub fn new(
        model: Box<dyn Layer>,
        optimizer: Box<dyn Optimizer>,
        loss_fn: Box<dyn Loss>,
    ) -> Self {
        Self {
            model,
            optimizer,
            loss_fn,
            scheduler: None,
            grad_clip: None,
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            patience: None,
            best_val_loss: f32::INFINITY,
            patience_counter: 0,
            checkpoint_dir: None,
        }
    }

    pub fn with_scheduler(mut self, scheduler: Box<dyn LRScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    pub fn with_early_stopping(mut self, patience: usize) -> Self {
        self.patience = Some(patience);
        self
    }

    pub fn with_grad_clip(mut self, clip: GradClip) -> Self {
        self.grad_clip = Some(clip);
        self
    }

    pub fn with_checkpoint_dir(mut self, dir: PathBuf) -> Self {
        self.checkpoint_dir = Some(dir);
        self
    }

    pub fn train_epoch(&mut self, train_loader: &mut DataLoader) -> f32 {
        self.model.train();
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        // DataLoader implements Iterator
        while let Some((inputs, targets)) = train_loader.next() {
            // Create a fresh tape for this step
            let mut tape = GradientTape::new();

            // Forward pass — tape records operations
            let outputs = self.model.forward(&inputs, Some(&mut tape));
            let loss = self.loss_fn.forward(&outputs, &targets, Some(&mut tape));

            // Backward pass — compute all gradients
            tape.backward(loss.id, &loss.shape);

            // Apply gradient clipping before optimizer step
            let param_ids: Vec<TensorId> = self.model.parameters()
                .iter().map(|p| p.id).collect();
            match &self.grad_clip {
                Some(GradClip::Norm(max_norm)) => {
                    clip_grad_norm(&mut tape, &param_ids, *max_norm);
                }
                Some(GradClip::Value(clip_value)) => {
                    clip_grad_value(&mut tape, &param_ids, *clip_value);
                }
                None => {}
            }

            // Optimizer step — reads gradients from tape
            let mut params: Vec<&mut Tensor> = self.model.parameters_mut();
            self.optimizer.step(&mut params.iter_mut().collect::<Vec<_>>(), &tape);

            total_loss += loss.item();
            num_batches += 1;
        }

        train_loader.reset();
        let avg_loss = total_loss / num_batches as f32;
        self.train_losses.push(avg_loss);

        avg_loss
    }

    pub fn validate(&mut self, val_loader: &mut DataLoader) -> f32 {
        self.model.eval();
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        // No tape needed for validation (no_grad)
        while let Some((inputs, targets)) = val_loader.next() {
            let outputs = self.model.forward(&inputs, None);
            let loss = self.loss_fn.forward(&outputs, &targets, None);

            total_loss += loss.item();
            num_batches += 1;
        }

        val_loader.reset();
        let avg_loss = total_loss / num_batches as f32;
        self.val_losses.push(avg_loss);

        // Early stopping check
        if let Some(_patience) = self.patience {
            if avg_loss < self.best_val_loss {
                self.best_val_loss = avg_loss;
                self.patience_counter = 0;
                if let Some(dir) = &self.checkpoint_dir {
                    let path = dir.join("best_model.bin");
                    save_checkpoint(&self.model, &path).ok();
                }
            } else {
                self.patience_counter += 1;
            }
        }

        avg_loss
    }

    pub fn fit(
        &mut self,
        train_loader: &mut DataLoader,
        val_loader: &mut DataLoader,
        num_epochs: usize,
    ) {
        for epoch in 0..num_epochs {
            let train_loss = self.train_epoch(train_loader);
            let val_loss = self.validate(val_loader);

            println!(
                "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}, lr={:.6}",
                epoch + 1,
                num_epochs,
                train_loss,
                val_loss,
                self.optimizer.lr(),
            );

            // Step scheduler
            if let Some(scheduler) = &mut self.scheduler {
                scheduler.step(&mut *self.optimizer);
            }

            // Early stopping
            if let Some(patience) = self.patience {
                if self.patience_counter >= patience {
                    println!("Early stopping triggered at epoch {}", epoch + 1);
                    break;
                }
            }
        }
    }

    /// Predict takes &mut self so it can switch model to eval mode.
    pub fn predict(&mut self, input: &Tensor) -> Tensor {
        self.model.eval();
        // No tape — pure inference
        self.model.forward(input, None)
    }
}
```

### Serialization

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    /// Serialized model parameter tensors (name -> raw bytes + shape + dtype).
    pub model_state: Vec<ParameterEntry>,

    /// Serialized optimizer state (moments, step count, etc.).
    pub optimizer_state: Vec<u8>,

    /// Training metadata.
    pub epoch: usize,
    pub best_val_loss: f32,
}

#[derive(Serialize, Deserialize)]
pub struct ParameterEntry {
    pub name: String,
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

pub fn save_checkpoint(model: &dyn Layer, path: &std::path::Path) -> std::io::Result<()> {
    let params = model.parameters();
    let entries: Vec<ParameterEntry> = params.iter().enumerate().map(|(i, p)| {
        ParameterEntry {
            name: format!("param_{}", i),
            data: p.to_bytes(),
            shape: p.shape.to_vec(),
            dtype: p.dtype,
        }
    }).collect();

    let checkpoint = Checkpoint {
        model_state: entries,
        optimizer_state: Vec::new(),
        epoch: 0,
        best_val_loss: f32::INFINITY,
    };

    let bytes = bincode::serialize(&checkpoint)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, bytes)
}

pub fn load_checkpoint(model: &mut dyn Layer, path: &std::path::Path) -> std::io::Result<()> {
    let bytes = std::fs::read(path)?;
    let checkpoint: Checkpoint = bincode::deserialize(&bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let mut params = model.parameters_mut();
    for (param, entry) in params.iter_mut().zip(checkpoint.model_state.iter()) {
        **param = Tensor::from_bytes(&entry.data, &entry.shape, entry.dtype);
    }

    Ok(())
}
```

### Metrics

```rust
pub struct Metrics {
    predictions: Vec<f32>,
    targets: Vec<f32>,
}

impl Metrics {
    pub fn new() -> Self {
        Self { predictions: Vec::new(), targets: Vec::new() }
    }

    pub fn update(&mut self, pred: &Tensor, target: &Tensor) {
        self.predictions.extend(pred.to_vec());
        self.targets.extend(target.to_vec());
    }

    pub fn mse(&self) -> f32 {
        self.predictions.iter()
            .zip(&self.targets)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>() / self.predictions.len() as f32
    }

    pub fn mae(&self) -> f32 {
        self.predictions.iter()
            .zip(&self.targets)
            .map(|(p, t)| (p - t).abs())
            .sum::<f32>() / self.predictions.len() as f32
    }

    pub fn rmse(&self) -> f32 {
        self.mse().sqrt()
    }

    pub fn r2_score(&self) -> f32 {
        let mean: f32 = self.targets.iter().sum::<f32>() / self.targets.len() as f32;
        let ss_tot: f32 = self.targets.iter().map(|t| (t - mean).powi(2)).sum();
        let ss_res: f32 = self.predictions.iter()
            .zip(&self.targets)
            .map(|(p, t)| (t - p).powi(2))
            .sum();

        1.0 - (ss_res / ss_tot)
    }

    pub fn mape(&self) -> f32 {
        let sum: f32 = self.predictions.iter()
            .zip(&self.targets)
            .map(|(p, t)| ((t - p) / t).abs())
            .sum();

        (sum / self.predictions.len() as f32) * 100.0
    }

    /// Symmetric Mean Absolute Percentage Error.
    pub fn smape(&self) -> f32 {
        let sum: f32 = self.predictions.iter()
            .zip(&self.targets)
            .map(|(p, t)| {
                let denom = (p.abs() + t.abs()) / 2.0;
                if denom == 0.0 { 0.0 } else { (p - t).abs() / denom }
            })
            .sum();

        (sum / self.predictions.len() as f32) * 100.0
    }

    pub fn reset(&mut self) {
        self.predictions.clear();
        self.targets.clear();
    }
}
```

---

## 10. Mixed-Precision Support

Store weights in BF16 to halve memory, but compute in F32 for numerical stability.
Gradients remain in F32.

```rust
pub struct MixedPrecisionConfig {
    /// Storage dtype for parameters (typically BF16).
    pub storage_dtype: DType,
    /// Compute dtype for forward/backward (typically F32).
    pub compute_dtype: DType,
    /// Gradient dtype (typically F32).
    pub grad_dtype: DType,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            storage_dtype: DType::BF16,
            compute_dtype: DType::F32,
            grad_dtype: DType::F32,
        }
    }
}

/// Mixed-precision Linear layer example.
pub struct MixedLinear {
    weight_bf16: Tensor,  // Stored in BF16
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl MixedLinear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Initialize in F32, then cast to BF16 for storage
        let weight_f32 = Tensor::randn(&[out_features, in_features])
            * (1.0 / in_features as f32).sqrt();
        let weight_bf16 = weight_f32.to_dtype(DType::BF16);

        let bias = if bias {
            Some(Tensor::zeros(&[out_features])) // bias stays F32
        } else {
            None
        };

        Self { weight_bf16, bias, in_features, out_features }
    }
}

impl Layer for MixedLinear {
    fn forward(
        &mut self,
        input: &Tensor,
        tape: Option<&mut GradientTape>,
    ) -> Tensor {
        // Upcast BF16 weight to F32 for computation
        let weight_f32 = self.weight_bf16.to_dtype(DType::F32);

        let output = matmul(input, &weight_f32.transpose(0, 1), tape);

        if let Some(bias) = &self.bias {
            add(&output, bias, tape)
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight_bf16];
        if let Some(bias) = &self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight_bf16];
        if let Some(bias) = &mut self.bias {
            params.push(bias);
        }
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
}
```

---

## 11. Performance Optimizations

### SIMD Operations

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn simd_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { simd_add_avx(a, b, result) }
            return;
        }
    }

    // Scalar fallback (also covers aarch64/NEON — add NEON intrinsics later)
    simd_add_scalar(a, b, result);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn simd_add_avx(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(offset), vr);
    }

    // Handle remaining elements
    for i in (chunks * 8)..len {
        result[i] = a[i] + b[i];
    }
}

fn simd_add_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}
```

### Parallel Processing with Rayon

```rust
use rayon::prelude::*;

pub fn parallel_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];

    let result: Vec<f32> = (0..m)
        .into_par_iter()
        .flat_map(|i| {
            (0..n).map(move |j| {
                (0..k).map(|p| a.get(&[i, p]) * b.get(&[p, j])).sum()
            })
        })
        .collect();

    Tensor::from_vec(result, &[m, n])
}
```

### TensorPool Integration with Backward Pass

The TensorPool (Section 2) integrates with the tape's backward pass to minimize
allocation overhead. When intermediate gradient tensors are no longer needed, their
backing buffers are returned to the pool. The `_raw` variants of ops (e.g.
`matmul_raw`, `add_raw`) used inside `BackwardOp::backward` allocate from and recycle
to the thread-local pool.

---

## 12. Usage Example

```rust
use timeseriesml::*;
use std::path::PathBuf;

fn main() {
    // Load OHLCV data
    let data = load_csv("btc_ohlcv.csv");

    // Create dataset with multi-target support
    let dataset = TimeSeriesDataset::new(data, 100, 1)
        .with_targets(TargetColumn::Multi(vec![
            "close".into(),
            "high".into(),
            "low".into(),
        ]));

    // Split train/val
    let (train_data, val_data) = dataset.split(0.8);

    // Create data loaders (implements Iterator)
    let mut train_loader = DataLoader::new(train_data, 32, true);
    let mut val_loader = DataLoader::new(val_data, 32, false);

    // Create model
    let mut model = TCN::new(
        5,                    // input_size: OHLCV features
        vec![64, 128, 256],   // num_channels
        3,                    // kernel_size
        0.2,                  // dropout
        3,                    // output_size: 3 targets
    );

    // Print model summary
    model_summary(&model);

    // Create optimizer (no longer takes parameters at construction)
    let optimizer = AdamW::new(
        0.001,            // lr
        (0.9, 0.999),     // betas
        1e-8,             // eps
        1e-5,             // weight_decay
    );

    // Warmup cosine scheduler
    let scheduler = WarmupCosineScheduler::new(
        0.001,  // base_lr
        10,     // warmup_steps (10 epochs)
        100,    // total_steps (100 epochs)
        1e-6,   // eta_min
    );

    // Create trainer with gradient clipping and checkpointing
    let mut trainer = Trainer::new(
        Box::new(model),
        Box::new(optimizer),
        Box::new(MSELoss),
    )
    .with_scheduler(Box::new(scheduler))
    .with_early_stopping(10)
    .with_grad_clip(GradClip::Norm(1.0))
    .with_checkpoint_dir(PathBuf::from("./checkpoints"));

    // Train
    trainer.fit(&mut train_loader, &mut val_loader, 100);

    // Predict (&mut self — can switch to eval mode)
    let test_input = Tensor::randn(&[1, 100, 5]);
    let prediction = trainer.predict(&test_input);

    println!("Prediction: {:?}", prediction);

    // Save final checkpoint
    save_checkpoint(&*trainer.model, &PathBuf::from("./checkpoints/final.bin")).ok();
}
```

---

## 13. Summary

This framework provides:

1. **Core tensor system** — `TensorId`-tracked tensors, `Arc<Storage>` (no cycles), `SmallVec` shapes, `TensorPool` buffer recycling
2. **Tape-based autodiff** — `GradientTape` with `BackwardOp` trait, `matmul_raw` non-tracking variant, `no_grad` support
3. **Neural network layers** — `Layer::forward(&mut self, ...)`, `Sequential` container, `parameter_count()`, `model_summary()`
4. **Optimizers** — SGD, Adam, AdamW reading gradients from tape; `GradClip::Norm` / `GradClip::Value`
5. **LR schedulers** — StepLR, CosineAnnealingLR, `WarmupCosineScheduler` (linear warmup + cosine decay)
6. **Loss functions** — MSE, MAE, Huber, CrossEntropy, Quantile (all tape-aware)
7. **Time series utilities** — OHLCV, multi-target `TargetColumn`, RSI with Wilder's EMA, `DataLoader` implementing `Iterator`
8. **Model architectures** — TCN (correct causal padding), Transformer (with `causal_mask`/`padding_mask`/`combined_mask`), N-BEATS (backcast subtraction)
9. **Training infrastructure** — `Trainer::predict(&mut self, ...)`, early stopping, gradient clipping
10. **Serialization** — `Checkpoint` struct with bincode, `save_checkpoint` / `load_checkpoint`
11. **Mixed precision** — `MixedPrecisionConfig` with BF16 storage / F32 compute
12. **Performance** — SIMD (AVX + NEON note), Rayon parallelism, TensorPool integration

### Issue Resolution Table

| # | Issue | Resolution |
|---|-------|------------|
| 1 | `saved_tensors()` returns `&[]` — graph has no edges | Tape-based autodiff; `TapeEntry` owns `saved_tensors: SmallVec<[Tensor; 2]>` |
| 2 | `MatMulBackward::backward` recurses infinitely | `matmul_raw()` non-tracking variant used in backward ops |
| 3 | `BatchNorm1d::forward` mutates through `&self` | `Layer::forward(&mut self, ...)` throughout |
| 4 | `Trainer::predict` calls `eval()` through `&self` | `predict(&mut self, ...)` |
| 5 | N-BEATS backcast residual not implemented | `forward_block()` returns `(backcast, forecast)`; `backcast = sub(&backcast, &block_backcast, tape)` |
| 6 | He init formula wrong | `(2.0 / in_features as f32).sqrt()` — correct formula |
| 7 | TCN causal padding wrong | `(kernel_size - 1) * dilation` full left pad + trim output |
| 8 | RSI uses simple average instead of EMA | Wilder's exponential smoothing with `alpha = 1/period` |
| 9 | No `no_grad` / inference context | `tape.enabled = false` or pass `None` as tape |
| 10 | Tensor owns grad by value | Grads live on `GradientTape` in `HashMap<TensorId, Tensor>` |
| 11 | `Storage::View` creates `Arc<Tensor>` cycles | `View` holds `Arc<Storage>` not `Arc<Tensor>` |
| 12 | No `Sequential` container | Full `Sequential` struct implementing `Layer` |
| 13 | No serialization design | `Checkpoint` struct with bincode + `save_checkpoint` / `load_checkpoint` |
| 14 | `DataLoader` doesn't implement `Iterator` | `impl Iterator for DataLoader` with `type Item = (Tensor, Tensor)` |
| 15 | No multi-feature target support | `TargetColumn` enum: `Single`, `Multi`, `Close` |
| 16 | No tensor memory pool | `TensorPool` with thread-local buffer recycling |
| 17 | No gradient clipping | `clip_grad_norm()`, `clip_grad_value()`, `GradClip` enum |
| 18 | No mixed-precision support | `MixedPrecisionConfig` with BF16 storage / F32 compute |
| 19 | No einsum implementation | Design note: contraction-path approach, common patterns as named ops first |
| 20 | No attention masking | `causal_mask()`, `padding_mask()`, `combined_mask()` helpers |
| 21 | No model summary / parameter counting | `parameter_count()` on Layer trait + `model_summary()` fn |
| 22 | No warmup scheduler | `WarmupCosineScheduler` with linear warmup + cosine decay |
