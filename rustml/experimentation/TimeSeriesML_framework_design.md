# TimeSeriesML Framework Design
## A from-scratch Rust ML framework optimized for time series forecasting

---

## Design Philosophy

**Goals:**
1. Pure Rust implementation with minimal dependencies
2. Optimized for CPU performance (SIMD, cache-friendly)
3. Time series first-class citizen (OHLCV, sequential data)
4. Type-safe, compile-time shape checking where possible
5. Educational but production-capable
6. Modular architecture for extensibility

**Non-Goals:**
- GPU support (initially - can be added later)
- Distributed training
- Complete feature parity with PyTorch

---

## Architecture Overview

```
timeseriesml/
├── core/           # Tensor operations and autodiff
├── nn/             # Neural network layers
├── optim/          # Optimizers
├── data/           # Data loading and preprocessing
├── timeseries/     # Time series specific components
├── models/         # Pre-built model architectures
└── training/       # Training infrastructure
```

---

## Module 1: Core Tensor System

### Tensor Structure

```rust
pub struct Tensor {
    // Data storage
    data: Storage,
    
    // Shape and strides
    shape: Vec<usize>,
    strides: Vec<usize>,
    
    // Gradient information
    grad: Option<Box<Tensor>>,
    requires_grad: bool,
    
    // Computational graph
    grad_fn: Option<Arc<dyn GradientFunction>>,
    
    // Metadata
    device: Device,  // CPU (GPU later)
    dtype: DType,    // f32, f64, i32, etc.
}

enum Storage {
    Owned(Vec<f32>),
    View { parent: Arc<Tensor>, offset: usize },
}

pub enum Device {
    CPU,
    // Future: GPU, TPU
}

pub enum DType {
    F32,
    F64,
    I32,
    // More types as needed
}
```

### Core Operations

**Creation Operations:**
- `Tensor::zeros(shape)` - Create tensor filled with zeros
- `Tensor::ones(shape)` - Create tensor filled with ones
- `Tensor::randn(shape)` - Random normal distribution
- `Tensor::uniform(shape, low, high)` - Uniform random
- `Tensor::from_vec(data, shape)` - From raw data
- `Tensor::arange(start, end, step)` - Range tensor

**Shape Operations:**
- `reshape(new_shape)` - Change shape (view if possible)
- `transpose(dim0, dim1)` - Swap dimensions
- `permute(dims)` - Reorder dimensions
- `squeeze(dim)` - Remove dimension of size 1
- `unsqueeze(dim)` - Add dimension of size 1
- `flatten()` - Flatten to 1D
- `view(shape)` - Zero-copy reshape

**Indexing & Slicing:**
- `slice(ranges)` - Multi-dimensional slicing
- `index_select(dim, indices)` - Select along dimension
- `gather(dim, indices)` - Gather values
- `masked_select(mask)` - Boolean indexing

**Math Operations:**
- Element-wise: `add, sub, mul, div, pow, sqrt, exp, log, abs`
- Reduction: `sum, mean, max, min, std, var`
- `matmul(other)` - Matrix multiplication
- `dot(other)` - Dot product
- Broadcasting support for all operations

**Advanced Operations:**
- `conv1d(weight, bias, stride, padding)` - 1D convolution
- `einsum(equation, tensors)` - Einstein summation
- `concat(tensors, dim)` - Concatenate tensors
- `split(sizes, dim)` - Split tensor
- `stack(tensors, dim)` - Stack tensors

### Memory Layout

```rust
// Efficient stride calculation for cache locality
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Index calculation with strides
fn compute_offset(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter()
        .zip(strides.iter())
        .map(|(i, s)| i * s)
        .sum()
}
```

---

## Module 2: Automatic Differentiation

### Gradient Function Trait

```rust
pub trait GradientFunction: Send + Sync {
    /// Compute gradients for inputs
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
    
    /// Get saved tensors for backward pass
    fn saved_tensors(&self) -> &[Arc<Tensor>];
}

// Example: Addition gradient function
struct AddBackward {
    left_shape: Vec<usize>,
    right_shape: Vec<usize>,
}

impl GradientFunction for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // Handle broadcasting in backward pass
        vec![
            unbroadcast(grad_output, &self.left_shape),
            unbroadcast(grad_output, &self.right_shape),
        ]
    }
    
    fn saved_tensors(&self) -> &[Arc<Tensor>] {
        &[]
    }
}
```

### Computational Graph

```rust
pub struct ComputeGraph {
    // Topologically sorted nodes
    nodes: Vec<Arc<Tensor>>,
}

impl ComputeGraph {
    pub fn backward(&self, loss: &Tensor) {
        // Initialize loss gradient
        loss.grad = Some(Box::new(Tensor::ones_like(loss)));
        
        // Reverse topological order
        for node in self.nodes.iter().rev() {
            if let Some(grad_fn) = &node.grad_fn {
                let grad_output = node.grad.as_ref().unwrap();
                let grads = grad_fn.backward(grad_output);
                
                // Accumulate gradients to inputs
                for (input, grad) in grad_fn.saved_tensors().iter().zip(grads) {
                    input.accumulate_grad(&grad);
                }
            }
        }
    }
}
```

### Gradient Operations

Each operation needs forward and backward:

```rust
// Example: MatMul
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // Forward pass
    let result = matmul_forward(a, b);
    
    // Setup backward
    if a.requires_grad || b.requires_grad {
        result.grad_fn = Some(Arc::new(MatMulBackward {
            a: Arc::new(a.clone()),
            b: Arc::new(b.clone()),
        }));
    }
    
    result
}

struct MatMulBackward {
    a: Arc<Tensor>,
    b: Arc<Tensor>,
}

impl GradientFunction for MatMulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        vec![
            matmul(grad_output, &self.b.transpose(-1, -2)),  // grad_a
            matmul(&self.a.transpose(-1, -2), grad_output),  // grad_b
        ]
    }
    
    fn saved_tensors(&self) -> &[Arc<Tensor>] {
        // Return references to saved tensors
        // Used by ComputeGraph for gradient flow
        &[]
    }
}
```

---

## Module 3: Neural Network Layers

### Layer Trait

```rust
pub trait Layer: Send + Sync {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Tensor;
    
    /// Get trainable parameters
    fn parameters(&self) -> Vec<&Tensor>;
    
    /// Get mutable parameters for optimizer
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    
    /// Training mode vs eval mode
    fn train(&mut self);
    fn eval(&mut self);
}

pub trait Module: Layer {
    /// Zero gradients
    fn zero_grad(&mut self);
    
    /// Get name for debugging
    fn name(&self) -> &str;
}
```

### Linear Layer

```rust
pub struct Linear {
    weight: Tensor,  // [out_features, in_features]
    bias: Option<Tensor>,  // [out_features]
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = Tensor::randn(&[out_features, in_features]) * 
                     (2.0 / (in_features as f32).sqrt());  // He initialization
        
        let bias = if bias {
            Some(Tensor::zeros(&[out_features]))
        } else {
            None
        };
        
        Self { weight, bias, in_features, out_features }
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, in_features]
        // output: [batch, out_features]
        let output = input.matmul(&self.weight.transpose(0, 1));
        
        if let Some(bias) = &self.bias {
            output + bias  // Broadcasting
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
    
    pub fn forward(&self, input: &Tensor, state: Option<(Tensor, Tensor)>) 
        -> (Tensor, (Tensor, Tensor)) 
    {
        // input: [seq_len, batch, input_size]
        // h_0: [num_layers, batch, hidden_size]
        // c_0: [num_layers, batch, hidden_size]
        
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
                );
                
                h = h.index_copy(layer, &h_new);
                c = c.index_copy(layer, &c_new);
            }
            
            outputs.push(h.slice(&[-1.., .., ..]));
        }
        
        let output = Tensor::stack(&outputs, 0);
        (output, (h, c))
    }
    
    fn lstm_cell(&self, x: &Tensor, h: &Tensor, c: &Tensor, layer: usize) 
        -> (Tensor, Tensor) 
    {
        // Gates: input, forget, cell, output
        let gates = x.matmul(&self.weights_ih[layer].t()) + &self.bias_ih[layer] +
                    h.matmul(&self.weights_hh[layer].t()) + &self.bias_hh[layer];
        
        let chunk_size = self.hidden_size;
        let i = gates.slice(&[.., 0..chunk_size]).sigmoid();
        let f = gates.slice(&[.., chunk_size..2*chunk_size]).sigmoid();
        let g = gates.slice(&[.., 2*chunk_size..3*chunk_size]).tanh();
        let o = gates.slice(&[.., 3*chunk_size..4*chunk_size]).sigmoid();
        
        let c_new = &f * c + &i * &g;
        let h_new = &o * c_new.tanh();
        
        (h_new, c_new)
    }
}
```

### Conv1D Layer

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
    ) -> Self {
        let weight = Tensor::randn(&[out_channels, in_channels, kernel_size]) *
                     (2.0 / (in_channels * kernel_size) as f32).sqrt();
        let bias = Some(Tensor::zeros(&[out_channels]));
        
        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation: 1,
        }
    }
}

impl Layer for Conv1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, in_channels, length]
        // output: [batch, out_channels, length_out]
        input.conv1d(&self.weight, self.bias.as_ref(), self.stride, self.padding)
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
    fn forward(&self, input: &Tensor) -> Tensor {
        input.maximum(&Tensor::zeros_like(input))
    }
    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

pub struct Tanh;
impl Layer for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.tanh()
    }
    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

pub struct Sigmoid;
impl Layer for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        (input.neg().exp() + 1.0).recip()
    }
    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

pub struct GELU;
impl Layer for GELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        // GELU(x) = x * Φ(x) where Φ is CDF of standard normal
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let coeff = (2.0 / std::f32::consts::PI).sqrt();
        let inner = (input + input.pow(3.0) * 0.044715) * coeff;
        input * 0.5 * (inner.tanh() + 1.0)
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
    normalized_shape: Vec<usize>,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Self {
        let weight = Tensor::ones(&normalized_shape);
        let bias = Tensor::zeros(&normalized_shape);
        Self { normalized_shape, weight, bias, eps }
    }
}

impl Layer for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mean = input.mean(&[-1], true);
        let var = input.var(&[-1], true);
        let normalized = (input - &mean) / (var + self.eps).sqrt();
        &normalized * &self.weight + &self.bias
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
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.training {
            let mean = input.mean(&[0], false);
            let var = input.var(&[0], false);
            
            // Update running statistics
            // self.running_mean = (1 - momentum) * running_mean + momentum * mean
            // self.running_var = (1 - momentum) * running_var + momentum * var
            
            let normalized = (input - &mean) / (var + self.eps).sqrt();
            &normalized * &self.weight + &self.bias
        } else {
            let normalized = (input - &self.running_mean) / 
                           (&self.running_var + self.eps).sqrt();
            &normalized * &self.weight + &self.bias
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
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.training && self.p > 0.0 {
            let mask = Tensor::rand_like(input).greater_than(self.p);
            (input * &mask) / (1.0 - self.p)
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

## Module 4: Optimizers

### Optimizer Trait

```rust
pub trait Optimizer {
    /// Perform single optimization step
    fn step(&mut self);
    
    /// Zero all gradients
    fn zero_grad(&mut self);
    
    /// Get current learning rate
    fn lr(&self) -> f32;
    
    /// Set learning rate
    fn set_lr(&mut self, lr: f32);
}

pub struct OptimizerConfig {
    parameters: Vec<Tensor>,
    lr: f32,
}
```

### SGD

```rust
pub struct SGD {
    parameters: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: Vec<Tensor>,
}

impl SGD {
    pub fn new(
        parameters: Vec<Tensor>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
    ) -> Self {
        let velocity = parameters.iter()
            .map(|p| Tensor::zeros_like(p))
            .collect();
        
        Self {
            parameters,
            lr,
            momentum,
            dampening: 0.0,
            weight_decay,
            nesterov: false,
            velocity,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, param) in self.parameters.iter_mut().enumerate() {
            if let Some(grad) = &param.grad {
                let mut d_p = grad.clone();
                
                // Weight decay
                if self.weight_decay != 0.0 {
                    d_p = &d_p + param * self.weight_decay;
                }
                
                // Momentum
                if self.momentum != 0.0 {
                    let v = &self.velocity[i];
                    self.velocity[i] = v * self.momentum + &d_p * (1.0 - self.dampening);
                    
                    if self.nesterov {
                        d_p = &d_p + &self.velocity[i] * self.momentum;
                    } else {
                        d_p = self.velocity[i].clone();
                    }
                }
                
                // Update parameter
                *param = param - &d_p * self.lr;
            }
        }
    }
    
    fn zero_grad(&mut self) {
        for param in &mut self.parameters {
            param.grad = None;
        }
    }
    
    fn lr(&self) -> f32 {
        self.lr
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}
```

### Adam

```rust
pub struct Adam {
    parameters: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    
    // State
    step_count: usize,
    m: Vec<Tensor>,  // First moment
    v: Vec<Tensor>,  // Second moment
}

impl Adam {
    pub fn new(
        parameters: Vec<Tensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let m = parameters.iter().map(|p| Tensor::zeros_like(p)).collect();
        let v = parameters.iter().map(|p| Tensor::zeros_like(p)).collect();
        
        Self {
            parameters,
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            weight_decay,
            step_count: 0,
            m,
            v,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.step_count += 1;
        
        for (i, param) in self.parameters.iter_mut().enumerate() {
            if let Some(grad) = &param.grad {
                let mut g = grad.clone();
                
                // Weight decay
                if self.weight_decay != 0.0 {
                    g = &g + param * self.weight_decay;
                }
                
                // Update biased first moment estimate
                self.m[i] = &self.m[i] * self.beta1 + &g * (1.0 - self.beta1);
                
                // Update biased second raw moment estimate
                self.v[i] = &self.v[i] * self.beta2 + g.pow(2.0) * (1.0 - self.beta2);
                
                // Bias correction
                let m_hat = &self.m[i] / (1.0 - self.beta1.powi(self.step_count as i32));
                let v_hat = &self.v[i] / (1.0 - self.beta2.powi(self.step_count as i32));
                
                // Update parameters
                *param = param - &m_hat * self.lr / (v_hat.sqrt() + self.eps);
            }
        }
    }
    
    fn zero_grad(&mut self) {
        for param in &mut self.parameters {
            param.grad = None;
        }
    }
    
    fn lr(&self) -> f32 {
        self.lr
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}
```

### AdamW

```rust
pub struct AdamW {
    parameters: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    
    step_count: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    pub fn new(
        parameters: Vec<Tensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let m = parameters.iter().map(|p| Tensor::zeros_like(p)).collect();
        let v = parameters.iter().map(|p| Tensor::zeros_like(p)).collect();
        
        Self {
            parameters,
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            weight_decay,
            step_count: 0,
            m,
            v,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.step_count += 1;
        
        for (i, param) in self.parameters.iter_mut().enumerate() {
            if let Some(grad) = &param.grad {
                // Update biased first moment estimate
                self.m[i] = &self.m[i] * self.beta1 + grad * (1.0 - self.beta1);
                
                // Update biased second raw moment estimate
                self.v[i] = &self.v[i] * self.beta2 + grad.pow(2.0) * (1.0 - self.beta2);
                
                // Bias correction
                let m_hat = &self.m[i] / (1.0 - self.beta1.powi(self.step_count as i32));
                let v_hat = &self.v[i] / (1.0 - self.beta2.powi(self.step_count as i32));
                
                // Update parameters with decoupled weight decay
                *param = param * (1.0 - self.lr * self.weight_decay) - 
                        &m_hat * self.lr / (v_hat.sqrt() + self.eps);
            }
        }
    }
    
    fn zero_grad(&mut self) {
        for param in &mut self.parameters {
            param.grad = None;
        }
    }
    
    fn lr(&self) -> f32 {
        self.lr
    }
    
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
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
```

---

## Module 5: Loss Functions

```rust
pub trait Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}

pub struct MSELoss;

impl Loss for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        (predictions - targets).pow(2.0).mean(&[], false)
    }
}

pub struct MAELoss;

impl Loss for MAELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        (predictions - targets).abs().mean(&[], false)
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
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let diff = (predictions - targets).abs();
        let quadratic = diff.pow(2.0) * 0.5;
        let linear = &diff * self.delta - self.delta * self.delta * 0.5;
        
        // Use quadratic where |diff| <= delta, linear otherwise
        let mask = diff.less_equal(self.delta);
        mask * quadratic + (1.0 - mask) * linear
    }
}

pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn forward(&self, logits: &Tensor, targets: &Tensor) -> Tensor {
        // logits: [batch, num_classes]
        // targets: [batch] (class indices)
        
        // Numerically stable softmax
        let max_logits = logits.max(&[-1], true);
        let exp_logits = (logits - &max_logits).exp();
        let log_sum_exp = exp_logits.sum(&[-1], true).log() + max_logits;
        
        // Gather target logits
        let target_logits = logits.gather(-1, targets);
        
        // Cross entropy
        (log_sum_exp - target_logits).mean(&[], false)
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
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let errors = targets - predictions;
        let positive_errors = errors.maximum(&Tensor::zeros_like(&errors));
        let negative_errors = (-&errors).maximum(&Tensor::zeros_like(&errors));
        
        (positive_errors * self.quantile + negative_errors * (1.0 - self.quantile))
            .mean(&[], false)
    }
}
```

---

## Module 6: Time Series Components

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

pub struct TimeSeriesDataset {
    data: Vec<OHLCVCandle>,
    window_size: usize,
    prediction_horizon: usize,
    
    // Feature engineering
    features: Vec<String>,
    
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
            scaler: None,
        }
    }
    
    pub fn len(&self) -> usize {
        self.data.len().saturating_sub(self.window_size + self.prediction_horizon - 1)
    }
    
    pub fn get(&self, idx: usize) -> (Tensor, Tensor) {
        // Input: [window_size, num_features]
        let input_start = idx;
        let input_end = idx + self.window_size;
        
        let input_data: Vec<f32> = self.data[input_start..input_end]
            .iter()
            .flat_map(|candle| vec![
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            ])
            .collect();
        
        let input = Tensor::from_vec(
            input_data,
            &[self.window_size, 5]
        );
        
        // Target: next candle close price
        let target_idx = input_end + self.prediction_horizon - 1;
        let target = Tensor::from_vec(
            vec![self.data[target_idx].close],
            &[1]
        );
        
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

pub struct RSI {
    period: usize,
}

impl Feature for RSI {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        let mut gains = Vec::new();
        let mut losses = Vec::new();
        
        for window in data.windows(2) {
            let change = window[1].close - window[0].close;
            gains.push(change.max(0.0));
            losses.push((-change).max(0.0));
        }
        
        gains.windows(self.period)
            .zip(losses.windows(self.period))
            .map(|(g, l)| {
                let avg_gain: f32 = g.iter().sum::<f32>() / self.period as f32;
                let avg_loss: f32 = l.iter().sum::<f32>() / self.period as f32;
                
                if avg_loss == 0.0 {
                    100.0
                } else {
                    100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
                }
            })
            .collect()
    }
    fn name(&self) -> &str { "rsi" }
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

### Normalization

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
            let transformed = match self.scaler_type {
                ScalerType::MinMax => (feature - p1) / p2,
                ScalerType::StandardScaler => (feature - p1) / p2,
                ScalerType::RobustScaler => (feature - p1) / p2,
            };
            result = result.index_copy(feat_idx, &transformed);
        }
        
        result
    }
    
    pub fn inverse_transform(&self, data: &Tensor) -> Tensor {
        let mut result = data.clone();
        
        for (feat_idx, (p1, p2)) in self.params.iter().enumerate() {
            let feature = data.slice(&[.., feat_idx..feat_idx+1]);
            let original = match self.scaler_type {
                ScalerType::MinMax => feature * p2 + p1,
                ScalerType::StandardScaler => feature * p2 + p1,
                ScalerType::RobustScaler => feature * p2 + p1,
            };
            result = result.index_copy(feat_idx, &original);
        }
        
        result
    }
}
```

### Data Loaders

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
    
    pub fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
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
    
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }
}
```

---

## Module 7: Model Architectures

### TCN (Temporal Convolutional Network)

```rust
pub struct TCNBlock {
    conv1: Conv1d,
    conv2: Conv1d,
    downsample: Option<Conv1d>,
    relu: ReLU,
    dropout: Dropout,
}

impl TCNBlock {
    pub fn new(
        n_inputs: usize,
        n_outputs: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        dropout: f32,
    ) -> Self {
        let padding = (kernel_size - 1) * dilation / 2;
        
        let conv1 = Conv1d::new(n_inputs, n_outputs, kernel_size, stride, padding);
        let conv2 = Conv1d::new(n_outputs, n_outputs, kernel_size, 1, padding);
        
        let downsample = if n_inputs != n_outputs {
            Some(Conv1d::new(n_inputs, n_outputs, 1, 1, 0))
        } else {
            None
        };
        
        Self {
            conv1,
            conv2,
            downsample,
            relu: ReLU,
            dropout: Dropout::new(dropout),
        }
    }
}

impl Layer for TCNBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = self.conv1.forward(input);
        out = self.relu.forward(&out);
        out = self.dropout.forward(&out);
        
        out = self.conv2.forward(&out);
        out = self.relu.forward(&out);
        out = self.dropout.forward(&out);
        
        let res = if let Some(downsample) = &self.downsample {
            downsample.forward(input)
        } else {
            input.clone()
        };
        
        self.relu.forward(&(out + res))
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
        if let Some(d) = &mut self.downsample {
            d.train();
        }
    }
    
    fn eval(&mut self) {
        self.conv1.eval();
        self.conv2.eval();
        self.dropout.eval();
        if let Some(d) = &mut self.downsample {
            d.eval();
        }
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
                1,
                dilation,
                dropout,
            ));
            in_channels = out_channels;
        }
        
        let fc = Linear::new(num_channels.last().unwrap().clone(), output_size, true);
        
        Self { blocks, fc }
    }
}

impl Layer for TCN {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for block in &self.blocks {
            out = block.forward(&out);
        }
        
        // Global average pooling
        out = out.mean(&[2], false);
        
        self.fc.forward(&out)
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
        for block in &mut self.blocks {
            block.train();
        }
        self.fc.train();
    }
    
    fn eval(&mut self) {
        for block in &mut self.blocks {
            block.eval();
        }
        self.fc.eval();
    }
}
```

### Transformer for Time Series

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
    
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];
        
        // Project and reshape to [batch, heads, seq_len, head_dim]
        let q = self.q_proj.forward(query)
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        
        let k = self.k_proj.forward(key)
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        
        let v = self.v_proj.forward(value)
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        
        // Attention scores
        let scores = q.matmul(&k.transpose(-2, -1)) / (self.head_dim as f32).sqrt();
        
        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            scores + mask * -1e9
        } else {
            scores
        };
        
        // Softmax
        let attn_weights = scores.softmax(-1);
        
        // Apply attention to values
        let out = attn_weights.matmul(&v);
        
        // Reshape and project
        let out = out.transpose(1, 2)
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim]);
        
        self.out_proj.forward(&out)
    }
}

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
}

impl Layer for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Self-attention with residual
        let attn_out = self.attention.forward(input, input, input, None);
        let attn_out = self.dropout.forward(&attn_out);
        let x = self.norm1.forward(&(input + attn_out));
        
        // Feed-forward with residual
        let ffn_out = self.ffn.forward(&x);
        let ffn_out = self.dropout.forward(&ffn_out);
        self.norm2.forward(&(&x + ffn_out))
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.ffn.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.ffn.parameters_mut());
        params
    }
    
    fn train(&mut self) {
        self.dropout.train();
    }
    
    fn eval(&mut self) {
        self.dropout.eval();
    }
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
        
        Self {
            embedding,
            positional_encoding,
            blocks,
            head,
        }
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
    fn forward(&self, input: &Tensor) -> Tensor {
        let seq_len = input.shape()[1];
        
        // Embed input
        let mut x = self.embedding.forward(input);
        
        // Add positional encoding
        let pos_enc = self.positional_encoding.slice(&[0..seq_len, ..]);
        x = x + pos_enc;
        
        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }
        
        // Take last timestep and project
        let last = x.slice(&[.., -1.., ..]).squeeze(1);
        self.head.forward(&last)
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
        for block in &mut self.blocks {
            block.train();
        }
    }
    
    fn eval(&mut self) {
        for block in &mut self.blocks {
            block.eval();
        }
    }
}
```

### N-BEATS

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
        
        fc_stack.push(Linear::new(input_size, layer_size, true));
        for _ in 1..num_layers {
            fc_stack.push(Linear::new(layer_size, layer_size, true));
        }
        
        let theta_b = Linear::new(layer_size, theta_size, true);
        let theta_f = Linear::new(layer_size, theta_size, true);
        
        Self {
            fc_stack,
            theta_b,
            theta_f,
            backcast_size,
            forecast_size,
        }
    }
}

impl Layer for NBeatsBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        
        // Fully connected stack with ReLU
        for fc in &self.fc_stack {
            x = fc.forward(&x);
            x = x.relu();
        }
        
        // Generate backcast and forecast
        let theta_b = self.theta_b.forward(&x);
        let theta_f = self.theta_f.forward(&x);
        
        // This is simplified - actual N-BEATS uses basis functions
        let backcast = theta_b;
        let forecast = theta_f;
        
        // Return both backcast and forecast
        // In actual usage, backcast is subtracted from input
        forecast
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for fc in &self.fc_stack {
            params.extend(fc.parameters());
        }
        params.extend(self.theta_b.parameters());
        params.extend(self.theta_f.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for fc in &mut self.fc_stack {
            params.extend(fc.parameters_mut());
        }
        params.extend(self.theta_b.parameters_mut());
        params.extend(self.theta_f.parameters_mut());
        params
    }
    
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

pub struct NBeats {
    stacks: Vec<Vec<NBeatsBlock>>,
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
        
        Self { stacks }
    }
}

impl Layer for NBeats {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut backcast = input.clone();
        let mut forecast = Tensor::zeros(&[input.shape()[0], self.stacks[0][0].forecast_size]);
        
        for stack in &self.stacks {
            for block in stack {
                let block_forecast = block.forward(&backcast);
                forecast = forecast + &block_forecast;
                // backcast = backcast - block_backcast (simplified)
            }
        }
        
        forecast
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for stack in &self.stacks {
            for block in stack {
                params.extend(block.parameters());
            }
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for stack in &mut self.stacks {
            for block in stack {
                params.extend(block.parameters_mut());
            }
        }
        params
    }
    
    fn train(&mut self) {
        for stack in &mut self.stacks {
            for block in stack {
                block.train();
            }
        }
    }
    
    fn eval(&mut self) {
        for stack in &mut self.stacks {
            for block in stack {
                block.eval();
            }
        }
    }
}
```

---

## Module 8: Training Infrastructure

```rust
pub struct Trainer {
    model: Box<dyn Layer>,
    optimizer: Box<dyn Optimizer>,
    loss_fn: Box<dyn Loss>,
    scheduler: Option<Box<dyn LRScheduler>>,
    
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
    
    pub fn train_epoch(&mut self, train_loader: &mut DataLoader) -> f32 {
        self.model.train();
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        while let Some((inputs, targets)) = train_loader.next_batch() {
            // Forward pass
            let outputs = self.model.forward(&inputs);
            let loss = self.loss_fn.forward(&outputs, &targets);
            
            // Backward pass
            self.optimizer.zero_grad();
            loss.backward();
            
            // Update weights
            self.optimizer.step();
            
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
        
        while let Some((inputs, targets)) = val_loader.next_batch() {
            let outputs = self.model.forward(&inputs);
            let loss = self.loss_fn.forward(&outputs, &targets);
            
            total_loss += loss.item();
            num_batches += 1;
        }
        
        val_loader.reset();
        let avg_loss = total_loss / num_batches as f32;
        self.val_losses.push(avg_loss);
        
        // Early stopping check
        if let Some(patience) = self.patience {
            if avg_loss < self.best_val_loss {
                self.best_val_loss = avg_loss;
                self.patience_counter = 0;
                self.save_checkpoint("best_model.pth");
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
                "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}",
                epoch + 1,
                num_epochs,
                train_loss,
                val_loss
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
    
    pub fn predict(&self, input: &Tensor) -> Tensor {
        self.model.eval();
        self.model.forward(input)
    }
    
    fn save_checkpoint(&self, filename: &str) {
        // Implement model serialization
        // This would save model parameters to disk
    }
    
    pub fn load_checkpoint(&mut self, filename: &str) {
        // Implement model deserialization
        // This would load model parameters from disk
    }
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
        Self {
            predictions: Vec::new(),
            targets: Vec::new(),
        }
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
    
    pub fn reset(&mut self) {
        self.predictions.clear();
        self.targets.clear();
    }
}
```

---

## Performance Optimizations

### SIMD Operations

```rust
// Use explicit SIMD for critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn simd_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { simd_add_avx(a, b, result) }
        } else {
            simd_add_scalar(a, b, result)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
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

### Parallel Processing

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

---

## Usage Example

```rust
use timeseriesml::*;

fn main() {
    // Load OHLCV data
    let data = load_csv("btc_ohlcv.csv");
    
    // Create dataset
    let dataset = TimeSeriesDataset::new(data, window_size=100, horizon=1);
    
    // Split train/val
    let (train_data, val_data) = dataset.split(0.8);
    
    // Create data loaders
    let mut train_loader = DataLoader::new(train_data, batch_size=32, shuffle=true);
    let mut val_loader = DataLoader::new(val_data, batch_size=32, shuffle=false);
    
    // Create model
    let model = TCN::new(
        input_size=5,  // OHLCV
        num_channels=vec![64, 128, 256],
        kernel_size=3,
        dropout=0.2,
        output_size=1,
    );
    
    // Create optimizer
    let optimizer = Adam::new(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    );
    
    // Create trainer
    let mut trainer = Trainer::new(
        Box::new(model),
        Box::new(optimizer),
        Box::new(MSELoss),
    )
    .with_scheduler(Box::new(CosineAnnealingLR::new(0.001, 100, 1e-6)))
    .with_early_stopping(10);
    
    // Train
    trainer.fit(&mut train_loader, &mut val_loader, num_epochs=100);
    
    // Predict
    let test_input = Tensor::randn(&[1, 100, 5]);
    let prediction = trainer.predict(&test_input);
    
    println!("Prediction: {:?}", prediction);
}
```

---

## Summary

This framework provides:

1. **Core tensor operations** with autodiff
2. **Neural network layers** (Linear, LSTM, Conv1D, etc.)
3. **Optimizers** (SGD, Adam, AdamW) with schedulers
4. **Loss functions** tailored for time series
5. **Time series utilities** (OHLCV handling, feature engineering, normalization)
6. **Model architectures** (TCN, Transformer, N-BEATS)
7. **Training infrastructure** with metrics and checkpointing
8. **Performance optimizations** (SIMD, parallelism)

Next steps would be implementing each module incrementally, starting with the core tensor system and autodiff engine.
