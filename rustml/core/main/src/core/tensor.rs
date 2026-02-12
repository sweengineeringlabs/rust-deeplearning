//! Tensor implementation with operations for neural network computations
//!
//! This module provides the core `Tensor` type with operations needed for
//! transformer models like GPT-2.

use crate::api::error::{TensorError, TensorResult};
use crate::core::shape::Shape;
use crate::api::types::Device;
use rand::Rng;
use std::fmt;
use std::sync::Arc;

/// A multi-dimensional array for numerical computations
#[derive(Clone)]
pub struct Tensor {
    data: Arc<Vec<f32>>,
    shape: Shape,
    strides: Vec<usize>,
    offset: usize,
    device: Device,
}

impl Tensor {
    // ==================== Constructors ====================

    /// Create a tensor from a vector with the given shape
    pub fn from_vec(data: Vec<f32>, shape: impl Into<Shape>) -> TensorResult<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: vec![data.len()],
            });
        }
        let strides = Self::compute_strides(&shape);
        Ok(Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let data = vec![0.0; shape.numel()];
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let data = vec![1.0; shape.numel()];
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        }
    }

    /// Create a tensor filled with a specific value
    pub fn full(shape: impl Into<Shape>, value: f32) -> Self {
        let shape = shape.into();
        let data = vec![value; shape.numel()];
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        }
    }

    /// Create a tensor with random values from standard normal distribution
    pub fn randn(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..shape.numel())
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = rng.r#gen::<f32>().max(1e-7);
                let u2: f32 = rng.r#gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        }
    }

    /// Create a tensor with random uniform values in [0, 1)
    pub fn rand(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..shape.numel()).map(|_| rng.r#gen()).collect();
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        }
    }

    /// Create an identity matrix
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let shape = Shape::new(vec![n, n]);
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        }
    }

    /// Create a lower triangular matrix of ones
    pub fn tril(size: usize) -> Self {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            for j in 0..=i {
                data[i * size + j] = 1.0;
            }
        }
        let shape = Shape::new(vec![size, size]);
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        }
    }

    /// Create an upper triangular matrix of ones
    pub fn triu(size: usize) -> Self {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            for j in i..size {
                data[i * size + j] = 1.0;
            }
        }
        let shape = Shape::new(vec![size, size]);
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        }
    }

    /// Create a 1D tensor with values from start to end (exclusive)
    pub fn arange(start: f32, end: f32, step: f32) -> TensorResult<Self> {
        if step == 0.0 {
            return Err(TensorError::InvalidOperation("step cannot be zero".into()));
        }
        let n = ((end - start) / step).ceil() as usize;
        let data: Vec<f32> = (0..n).map(|i| start + (i as f32) * step).collect();
        let shape = Shape::new(vec![n]);
        let strides = Self::compute_strides(&shape);
        Ok(Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            device: Device::Cpu,
        })
    }

    // ==================== Properties ====================

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the underlying data as a slice (contiguous tensors only)
    pub fn data(&self) -> TensorResult<&[f32]> {
        if self.is_contiguous() {
            Ok(&self.data[self.offset..self.offset + self.numel()])
        } else {
            Err(TensorError::InvalidOperation(
                "Cannot get data slice of non-contiguous tensor".into(),
            ))
        }
    }

    /// Convert to a Vec (always works, may copy)
    pub fn to_vec(&self) -> Vec<f32> {
        if self.is_contiguous() {
            self.data[self.offset..self.offset + self.numel()].to_vec()
        } else {
            self.iter().collect()
        }
    }

    /// Check if tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = Self::compute_strides(&self.shape);
        self.strides == expected_strides && self.offset == 0
    }

    /// Make tensor contiguous (copy if necessary)
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            let data: Vec<f32> = self.iter().collect();
            Self::from_vec(data, self.shape.clone()).unwrap()
        }
    }

    // ==================== Indexing Operations ====================

    /// Get element at flat index
    fn get_flat(&self, idx: usize) -> f32 {
        self.data[self.offset + idx]
    }

    /// Iterate over all elements
    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        TensorIterator::new(self)
    }

    /// Get a single element by indices
    pub fn get(&self, indices: &[usize]) -> TensorResult<f32> {
        if indices.len() != self.ndim() {
            return Err(TensorError::InvalidOperation(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }
        let mut offset = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape.dims()[i] {
                return Err(TensorError::IndexOutOfBounds {
                    dim: i,
                    index: idx,
                    size: self.shape.dims()[i],
                });
            }
            offset += idx * self.strides[i];
        }
        Ok(self.data[offset])
    }

    /// Slice the tensor along a dimension
    pub fn slice(&self, dim: i64, start: usize, end: usize) -> TensorResult<Self> {
        let dim_idx = self.normalize_dim(dim)?;
        let dim_size = self.shape.dims()[dim_idx];

        if start > end || end > dim_size {
            return Err(TensorError::InvalidSliceRange {
                start,
                end,
                size: dim_size,
            });
        }

        let new_size = end - start;
        let mut new_dims = self.shape.dims().to_vec();
        new_dims[dim_idx] = new_size;
        let new_shape = Shape::new(new_dims);

        // Collect the sliced data
        let mut new_data = Vec::with_capacity(new_shape.numel());
        self.collect_slice(&mut new_data, dim_idx, start, end, &[], 0);

        Self::from_vec(new_data, new_shape)
    }

    fn collect_slice(
        &self,
        result: &mut Vec<f32>,
        slice_dim: usize,
        start: usize,
        end: usize,
        indices: &[usize],
        depth: usize,
    ) {
        if depth == self.ndim() {
            if let Ok(val) = self.get(indices) {
                result.push(val);
            }
            return;
        }

        let range = if depth == slice_dim {
            start..end
        } else {
            0..self.shape.dims()[depth]
        };

        for i in range {
            let mut new_indices = indices.to_vec();
            new_indices.push(i);
            self.collect_slice(result, slice_dim, start, end, &new_indices, depth + 1);
        }
    }

    /// Select a single index along a dimension (reduces dimensionality)
    pub fn select(&self, dim: i64, index: usize) -> TensorResult<Self> {
        let dim_idx = self.normalize_dim(dim)?;
        let dim_size = self.shape.dims()[dim_idx];

        if index >= dim_size {
            return Err(TensorError::IndexOutOfBounds {
                dim: dim_idx,
                index,
                size: dim_size,
            });
        }

        // New shape without the selected dimension
        let mut new_dims: Vec<usize> = self.shape.dims().to_vec();
        new_dims.remove(dim_idx);
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(new_dims)
        };

        // Collect selected data
        let mut new_data = Vec::with_capacity(new_shape.numel());
        self.collect_select(&mut new_data, dim_idx, index, &[], 0);

        Self::from_vec(new_data, new_shape)
    }

    fn collect_select(
        &self,
        result: &mut Vec<f32>,
        select_dim: usize,
        select_idx: usize,
        indices: &[usize],
        depth: usize,
    ) {
        if depth == self.ndim() {
            if let Ok(val) = self.get(indices) {
                result.push(val);
            }
            return;
        }

        if depth == select_dim {
            let mut new_indices = indices.to_vec();
            new_indices.push(select_idx);
            self.collect_select(result, select_dim, select_idx, &new_indices, depth + 1);
        } else {
            for i in 0..self.shape.dims()[depth] {
                let mut new_indices = indices.to_vec();
                new_indices.push(i);
                self.collect_select(result, select_dim, select_idx, &new_indices, depth + 1);
            }
        }
    }

    /// Concatenate tensors along a dimension
    pub fn cat(tensors: &[&Tensor], dim: i64) -> TensorResult<Self> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyTensor);
        }

        let first = tensors[0];
        let dim_idx = first.normalize_dim(dim)?;

        // Validate shapes
        for t in tensors.iter().skip(1) {
            if t.ndim() != first.ndim() {
                return Err(TensorError::ShapeMismatch {
                    expected: first.shape.dims().to_vec(),
                    got: t.shape.dims().to_vec(),
                });
            }
            for (i, (&s1, &s2)) in first.shape.dims().iter().zip(t.shape.dims()).enumerate() {
                if i != dim_idx && s1 != s2 {
                    return Err(TensorError::ShapeMismatch {
                        expected: first.shape.dims().to_vec(),
                        got: t.shape.dims().to_vec(),
                    });
                }
            }
        }

        // Compute output shape
        let total_dim_size: usize = tensors.iter().map(|t| t.shape.dims()[dim_idx]).sum();
        let mut new_dims = first.shape.dims().to_vec();
        new_dims[dim_idx] = total_dim_size;
        let new_shape = Shape::new(new_dims);

        // Collect data
        let mut new_data = Vec::with_capacity(new_shape.numel());
        Self::collect_cat(&mut new_data, tensors, dim_idx, &[], 0);

        Self::from_vec(new_data, new_shape)
    }

    fn collect_cat(
        result: &mut Vec<f32>,
        tensors: &[&Tensor],
        cat_dim: usize,
        indices: &[usize],
        depth: usize,
    ) {
        let ndim = tensors[0].ndim();
        if depth == ndim {
            // We need to figure out which tensor and what index
            let cat_idx = indices[cat_dim];
            let mut offset = 0;
            for t in tensors {
                let dim_size = t.shape.dims()[cat_dim];
                if cat_idx < offset + dim_size {
                    let mut t_indices = indices.to_vec();
                    t_indices[cat_dim] = cat_idx - offset;
                    if let Ok(val) = t.get(&t_indices) {
                        result.push(val);
                    }
                    return;
                }
                offset += dim_size;
            }
            return;
        }

        let range = if depth == cat_dim {
            let total: usize = tensors.iter().map(|t| t.shape.dims()[cat_dim]).sum();
            0..total
        } else {
            0..tensors[0].shape.dims()[depth]
        };

        for i in range {
            let mut new_indices = indices.to_vec();
            new_indices.push(i);
            Self::collect_cat(result, tensors, cat_dim, &new_indices, depth + 1);
        }
    }

    /// Fill tensor where mask is true with specified value
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> TensorResult<Self> {
        // Broadcast shapes
        let broadcast_shape = self
            .shape
            .broadcast_with(&mask.shape)
            .ok_or_else(|| TensorError::BroadcastError {
                shape1: self.shape.dims().to_vec(),
                shape2: mask.shape.dims().to_vec(),
            })?;

        let self_broadcast = self.broadcast_to(&broadcast_shape)?;
        let mask_broadcast = mask.broadcast_to(&broadcast_shape)?;

        let new_data: Vec<f32> = self_broadcast
            .iter()
            .zip(mask_broadcast.iter())
            .map(|(v, m)| if m != 0.0 { value } else { v })
            .collect();

        Self::from_vec(new_data, broadcast_shape)
    }

    // ==================== Shape Operations ====================

    /// Reshape the tensor
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> TensorResult<Self> {
        let new_shape = new_shape.into();
        if new_shape.numel() != self.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: new_shape.dims().to_vec(),
                got: self.shape.dims().to_vec(),
            });
        }
        let contiguous = self.contiguous();
        let strides = Self::compute_strides(&new_shape);
        Ok(Self {
            data: contiguous.data,
            shape: new_shape,
            strides,
            offset: 0,
            device: self.device,
        })
    }

    /// Add a dimension of size 1 at the specified position
    pub fn unsqueeze(&self, dim: i64) -> TensorResult<Self> {
        let ndim = self.ndim() as i64 + 1;
        let normalized = if dim < 0 { dim + ndim } else { dim };
        if normalized < 0 || normalized > self.ndim() as i64 {
            return Err(TensorError::InvalidDimension {
                dim,
                ndim: self.ndim(),
            });
        }
        let idx = normalized as usize;
        let mut new_dims = self.shape.dims().to_vec();
        new_dims.insert(idx, 1);
        self.reshape(new_dims)
    }

    /// Remove a dimension of size 1
    pub fn squeeze(&self, dim: i64) -> TensorResult<Self> {
        let dim_idx = self.normalize_dim(dim)?;
        if self.shape.dims()[dim_idx] != 1 {
            return Err(TensorError::InvalidOperation(format!(
                "Cannot squeeze dimension {} with size {}",
                dim, self.shape.dims()[dim_idx]
            )));
        }
        let mut new_dims = self.shape.dims().to_vec();
        new_dims.remove(dim_idx);
        self.reshape(new_dims)
    }

    /// Transpose two dimensions
    pub fn transpose(&self, dim0: i64, dim1: i64) -> TensorResult<Self> {
        let dim0_idx = self.normalize_dim(dim0)?;
        let dim1_idx = self.normalize_dim(dim1)?;

        let mut new_dims = self.shape.dims().to_vec();
        new_dims.swap(dim0_idx, dim1_idx);
        let new_shape = Shape::new(new_dims);

        let mut perm: Vec<usize> = (0..self.ndim()).collect();
        perm.swap(dim0_idx, dim1_idx);

        self.permute(&perm)
    }

    /// Transpose last two dimensions (for matrix operations)
    pub fn t(&self) -> TensorResult<Self> {
        if self.ndim() < 2 {
            return Err(TensorError::InvalidOperation(
                "Cannot transpose tensor with less than 2 dimensions".into(),
            ));
        }
        self.transpose(-2, -1)
    }

    /// Permute dimensions
    pub fn permute(&self, dims: &[usize]) -> TensorResult<Self> {
        if dims.len() != self.ndim() {
            return Err(TensorError::InvalidOperation(format!(
                "Permutation must have {} dimensions",
                self.ndim()
            )));
        }

        let new_dims: Vec<usize> = dims.iter().map(|&d| self.shape.dims()[d]).collect();
        let new_shape = Shape::new(new_dims);

        let mut new_data = Vec::with_capacity(self.numel());
        self.collect_permute(&mut new_data, dims, &[], 0);

        Self::from_vec(new_data, new_shape)
    }

    fn collect_permute(
        &self,
        result: &mut Vec<f32>,
        perm: &[usize],
        new_indices: &[usize],
        depth: usize,
    ) {
        if depth == self.ndim() {
            // Convert new_indices back to original order
            let mut orig_indices = vec![0; self.ndim()];
            for (new_dim, &orig_dim) in perm.iter().enumerate() {
                orig_indices[orig_dim] = new_indices[new_dim];
            }
            if let Ok(val) = self.get(&orig_indices) {
                result.push(val);
            }
            return;
        }

        let orig_dim = perm[depth];
        for i in 0..self.shape.dims()[orig_dim] {
            let mut ni = new_indices.to_vec();
            ni.push(i);
            self.collect_permute(result, perm, &ni, depth + 1);
        }
    }

    /// Broadcast to a new shape
    pub fn broadcast_to(&self, shape: &Shape) -> TensorResult<Self> {
        if self.shape.dims() == shape.dims() {
            return Ok(self.clone());
        }

        // Check if broadcast is valid
        let broadcast_shape = self.shape.broadcast_with(shape).ok_or_else(|| {
            TensorError::BroadcastError {
                shape1: self.shape.dims().to_vec(),
                shape2: shape.dims().to_vec(),
            }
        })?;

        if broadcast_shape.dims() != shape.dims() {
            return Err(TensorError::BroadcastError {
                shape1: self.shape.dims().to_vec(),
                shape2: shape.dims().to_vec(),
            });
        }

        let mut new_data = Vec::with_capacity(shape.numel());
        self.collect_broadcast(&mut new_data, shape, &[], 0);

        Self::from_vec(new_data, shape.clone())
    }

    fn collect_broadcast(
        &self,
        result: &mut Vec<f32>,
        target_shape: &Shape,
        indices: &[usize],
        depth: usize,
    ) {
        if depth == target_shape.ndim() {
            // Map target indices to source indices
            let offset = target_shape.ndim() - self.ndim();
            let mut src_indices = Vec::with_capacity(self.ndim());
            for i in 0..self.ndim() {
                let target_idx = indices[offset + i];
                let src_size = self.shape.dims()[i];
                src_indices.push(if src_size == 1 { 0 } else { target_idx });
            }
            if let Ok(val) = self.get(&src_indices) {
                result.push(val);
            }
            return;
        }

        for i in 0..target_shape.dims()[depth] {
            let mut ni = indices.to_vec();
            ni.push(i);
            self.collect_broadcast(result, target_shape, &ni, depth + 1);
        }
    }

    // ==================== Math Operations ====================

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> TensorResult<Self> {
        self.binary_op(other, |a, b| a + b)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> TensorResult<Self> {
        self.binary_op(other, |a, b| a - b)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> TensorResult<Self> {
        self.binary_op(other, |a, b| a * b)
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> TensorResult<Self> {
        self.binary_op(other, |a, b| a / b)
    }

    /// Add a scalar
    pub fn add_scalar(&self, scalar: f32) -> Self {
        self.unary_op(|x| x + scalar)
    }

    /// Multiply by a scalar
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        self.unary_op(|x| x * scalar)
    }

    /// Divide by a scalar
    pub fn div_scalar(&self, scalar: f32) -> Self {
        self.unary_op(|x| x / scalar)
    }

    /// Negate
    pub fn neg(&self) -> Self {
        self.unary_op(|x| -x)
    }

    /// Square root
    pub fn sqrt(&self) -> Self {
        self.unary_op(|x| x.sqrt())
    }

    /// Exponential
    pub fn exp(&self) -> Self {
        self.unary_op(|x| x.exp())
    }

    /// Natural logarithm
    pub fn log(&self) -> Self {
        self.unary_op(|x| x.ln())
    }

    /// Power
    pub fn pow(&self, exp: f32) -> Self {
        self.unary_op(|x| x.powf(exp))
    }

    /// Absolute value
    pub fn abs(&self) -> Self {
        self.unary_op(|x| x.abs())
    }

    /// Clamp values to a range
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        self.unary_op(|x| x.clamp(min, max))
    }

    /// Rectified Linear Unit (ReLU)
    pub fn relu(&self) -> Self {
        self.unary_op(|x| x.max(0.0))
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Self {
        self.unary_op(|x| 1.0 / (1.0 + (-x).exp()))
    }

    /// Hyperbolic tangent
    pub fn tanh(&self) -> Self {
        self.unary_op(|x| x.tanh())
    }

    /// GELU activation (Gaussian Error Linear Unit)
    pub fn gelu(&self) -> Self {
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        self.unary_op(|x| {
            let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
        })
    }

    /// Softmax along a dimension
    pub fn softmax(&self, dim: i64) -> TensorResult<Self> {
        let dim_idx = self.normalize_dim(dim)?;

        // Subtract max for numerical stability
        let max_vals = self.max(dim)?.0;
        let max_broadcast = max_vals.unsqueeze(dim)?;
        let shifted = self.sub(&max_broadcast.broadcast_to(&self.shape)?)?;

        // exp and sum
        let exp_vals = shifted.exp();
        let sum_exp = exp_vals.sum(dim)?;
        let sum_broadcast = sum_exp.unsqueeze(dim)?;

        exp_vals.div(&sum_broadcast.broadcast_to(&self.shape)?)
    }

    // ==================== Reduction Operations ====================

    /// Sum all elements
    pub fn sum_all(&self) -> f32 {
        self.iter().sum()
    }

    /// Mean of all elements
    pub fn mean_all(&self) -> f32 {
        self.sum_all() / self.numel() as f32
    }

    /// Sum along a dimension
    pub fn sum(&self, dim: i64) -> TensorResult<Self> {
        self.reduce(dim, 0.0, |acc, x| acc + x)
    }

    /// Mean along a dimension
    pub fn mean(&self, dim: i64) -> TensorResult<Self> {
        let dim_idx = self.normalize_dim(dim)?;
        let dim_size = self.shape.dims()[dim_idx] as f32;
        let sum = self.sum(dim)?;
        Ok(sum.div_scalar(dim_size))
    }

    /// Variance along a dimension
    pub fn var(&self, dim: i64) -> TensorResult<Self> {
        let mean = self.mean(dim)?;
        let mean_broadcast = mean.unsqueeze(dim)?.broadcast_to(&self.shape)?;
        let diff = self.sub(&mean_broadcast)?;
        let sq_diff = diff.mul(&diff)?;
        sq_diff.mean(dim)
    }

    /// Max along a dimension
    pub fn max(&self, dim: i64) -> TensorResult<(Self, Self)> {
        let dim_idx = self.normalize_dim(dim)?;
        let dim_size = self.shape.dims()[dim_idx];

        let mut new_dims = self.shape.dims().to_vec();
        new_dims.remove(dim_idx);
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(new_dims)
        };

        let mut values = Vec::with_capacity(new_shape.numel());
        let mut indices = Vec::with_capacity(new_shape.numel());

        self.collect_max(&mut values, &mut indices, dim_idx, dim_size, &[], 0);

        Ok((
            Self::from_vec(values, new_shape.clone())?,
            Self::from_vec(indices, new_shape)?,
        ))
    }

    fn collect_max(
        &self,
        values: &mut Vec<f32>,
        indices: &mut Vec<f32>,
        reduce_dim: usize,
        dim_size: usize,
        current_indices: &[usize],
        depth: usize,
    ) {
        // For 1D tensors reducing along dim 0
        if self.ndim() == 1 && reduce_dim == 0 {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0usize;
            for i in 0..dim_size {
                if let Ok(val) = self.get(&[i]) {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
            }
            values.push(max_val);
            indices.push(max_idx as f32);
            return;
        }

        // We have all indices except reduce_dim, compute max
        if current_indices.len() == self.ndim() - 1 {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0usize;

            for i in 0..dim_size {
                let mut full_indices = Vec::with_capacity(self.ndim());
                let mut ci = 0;
                for d in 0..self.ndim() {
                    if d == reduce_dim {
                        full_indices.push(i);
                    } else {
                        full_indices.push(current_indices[ci]);
                        ci += 1;
                    }
                }
                if let Ok(val) = self.get(&full_indices) {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
            }
            values.push(max_val);
            indices.push(max_idx as f32);
            return;
        }

        // Calculate which dimension we're iterating over
        let current_dim = if depth >= reduce_dim { depth + 1 } else { depth };
        if current_dim >= self.ndim() {
            return;
        }

        for i in 0..self.shape.dims()[current_dim] {
            let mut ni = current_indices.to_vec();
            ni.push(i);
            self.collect_max(values, indices, reduce_dim, dim_size, &ni, depth + 1);
        }
    }

    /// Argmax along a dimension
    pub fn argmax(&self, dim: i64) -> TensorResult<Self> {
        let (_, indices) = self.max(dim)?;
        Ok(indices)
    }

    /// Min along a dimension
    pub fn min(&self, dim: i64) -> TensorResult<(Self, Self)> {
        // Similar to max but with min logic
        let negated = self.neg();
        let (max_vals, indices) = negated.max(dim)?;
        Ok((max_vals.neg(), indices))
    }

    // ==================== Matrix Operations ====================

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> TensorResult<Self> {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(TensorError::InvalidOperation(
                "Matrix multiplication requires at least 2D tensors".into(),
            ));
        }

        let self_shape = self.shape.dims();
        let other_shape = other.shape.dims();

        let m = self_shape[self_shape.len() - 2];
        let k = self_shape[self_shape.len() - 1];
        let k2 = other_shape[other_shape.len() - 2];
        let n = other_shape[other_shape.len() - 1];

        if k != k2 {
            return Err(TensorError::MatmulDimensionMismatch { left: k, right: k2 });
        }

        // Handle batch dimensions
        let self_batch: Vec<usize> = self_shape[..self_shape.len() - 2].to_vec();
        let other_batch: Vec<usize> = other_shape[..other_shape.len() - 2].to_vec();

        // Broadcast batch dimensions
        let batch_shape = if self_batch.is_empty() && other_batch.is_empty() {
            vec![]
        } else {
            let self_batch_shape = Shape::new(self_batch.clone());
            let other_batch_shape = Shape::new(other_batch.clone());
            self_batch_shape
                .broadcast_with(&other_batch_shape)
                .ok_or_else(|| TensorError::BroadcastError {
                    shape1: self_batch.clone(),
                    shape2: other_batch.clone(),
                })?
                .dims()
                .to_vec()
        };

        let mut result_shape = batch_shape.clone();
        result_shape.push(m);
        result_shape.push(n);

        let batch_numel: usize = batch_shape.iter().product::<usize>().max(1);
        let mut result_data = Vec::with_capacity(batch_numel * m * n);

        // Perform batched matmul
        for batch_idx in 0..batch_numel {
            // Get batch indices
            let mut remaining = batch_idx;
            let mut batch_indices = vec![0; batch_shape.len()];
            for i in (0..batch_shape.len()).rev() {
                batch_indices[i] = remaining % batch_shape[i];
                remaining /= batch_shape[i];
            }

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        let mut self_idx = batch_indices
                            .iter()
                            .enumerate()
                            .map(|(bi, &idx)| {
                                if bi < self_batch.len() {
                                    idx % self_shape[bi]
                                } else {
                                    0
                                }
                            })
                            .collect::<Vec<_>>();
                        while self_idx.len() < self_shape.len() - 2 {
                            self_idx.push(0);
                        }
                        self_idx.push(i);
                        self_idx.push(l);

                        let mut other_idx = batch_indices
                            .iter()
                            .enumerate()
                            .map(|(bi, &idx)| {
                                if bi < other_batch.len() {
                                    idx % other_shape[bi]
                                } else {
                                    0
                                }
                            })
                            .collect::<Vec<_>>();
                        while other_idx.len() < other_shape.len() - 2 {
                            other_idx.push(0);
                        }
                        other_idx.push(l);
                        other_idx.push(j);

                        let a = self.get(&self_idx).unwrap_or(0.0);
                        let b = other.get(&other_idx).unwrap_or(0.0);
                        sum += a * b;
                    }
                    result_data.push(sum);
                }
            }
        }

        Self::from_vec(result_data, result_shape)
    }

    // ==================== Utility Methods ====================

    fn compute_strides(shape: &Shape) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.ndim());
        let mut stride = 1;
        for &dim in shape.dims().iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();
        strides
    }

    fn normalize_dim(&self, dim: i64) -> TensorResult<usize> {
        self.shape.normalize_dim(dim).ok_or(TensorError::InvalidDimension {
            dim,
            ndim: self.ndim(),
        })
    }

    fn unary_op(&self, f: impl Fn(f32) -> f32) -> Self {
        let data: Vec<f32> = self.iter().map(f).collect();
        Self::from_vec(data, self.shape.clone()).unwrap()
    }

    fn binary_op(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> TensorResult<Self> {
        let broadcast_shape =
            self.shape
                .broadcast_with(&other.shape)
                .ok_or_else(|| TensorError::BroadcastError {
                    shape1: self.shape.dims().to_vec(),
                    shape2: other.shape.dims().to_vec(),
                })?;

        let self_broadcast = self.broadcast_to(&broadcast_shape)?;
        let other_broadcast = other.broadcast_to(&broadcast_shape)?;

        let data: Vec<f32> = self_broadcast
            .iter()
            .zip(other_broadcast.iter())
            .map(|(a, b)| f(a, b))
            .collect();

        Self::from_vec(data, broadcast_shape)
    }

    fn reduce(&self, dim: i64, init: f32, f: impl Fn(f32, f32) -> f32) -> TensorResult<Self> {
        let dim_idx = self.normalize_dim(dim)?;

        let mut new_dims = self.shape.dims().to_vec();
        new_dims.remove(dim_idx);
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(new_dims)
        };

        let dim_size = self.shape.dims()[dim_idx];
        let mut new_data = Vec::with_capacity(new_shape.numel());

        self.collect_reduce(&mut new_data, dim_idx, dim_size, init, &f, &[], 0);

        Self::from_vec(new_data, new_shape)
    }

    fn collect_reduce(
        &self,
        result: &mut Vec<f32>,
        reduce_dim: usize,
        dim_size: usize,
        init: f32,
        f: &impl Fn(f32, f32) -> f32,
        indices: &[usize],
        depth: usize,
    ) {
        if indices.len() == self.ndim() - 1 {
            // Reduce over reduce_dim
            let mut acc = init;
            for i in 0..dim_size {
                let mut full_indices = Vec::with_capacity(self.ndim());
                let mut idx_ptr = 0;
                for d in 0..self.ndim() {
                    if d == reduce_dim {
                        full_indices.push(i);
                    } else {
                        full_indices.push(indices[idx_ptr]);
                        idx_ptr += 1;
                    }
                }
                if let Ok(val) = self.get(&full_indices) {
                    acc = f(acc, val);
                }
            }
            result.push(acc);
            return;
        }

        let current_dim = if depth >= reduce_dim { depth + 1 } else { depth };
        if current_dim >= self.ndim() {
            return;
        }

        for i in 0..self.shape.dims()[current_dim] {
            let mut ni = indices.to_vec();
            ni.push(i);
            self.collect_reduce(result, reduce_dim, dim_size, init, f, &ni, depth + 1);
        }
    }
}

// ==================== Tensor Iterator ====================

struct TensorIterator<'a> {
    tensor: &'a Tensor,
    indices: Vec<usize>,
    done: bool,
}

impl<'a> TensorIterator<'a> {
    fn new(tensor: &'a Tensor) -> Self {
        let indices = vec![0; tensor.ndim()];
        let done = tensor.numel() == 0;
        Self { tensor, indices, done }
    }

    fn advance(&mut self) {
        for i in (0..self.indices.len()).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.tensor.shape.dims()[i] {
                return;
            }
            self.indices[i] = 0;
        }
        self.done = true;
    }
}

impl Iterator for TensorIterator<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let val = self.tensor.get(&self.indices).ok()?;
        self.advance();
        Some(val)
    }
}

// ==================== Display ====================

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, device={:?})", self.shape, self.device)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.numel() <= 100 {
            write!(f, "Tensor({}, {:?})", self.shape, self.to_vec())
        } else {
            write!(
                f,
                "Tensor({}, [{:.4}, {:.4}, ..., {:.4}, {:.4}])",
                self.shape,
                self.get(&vec![0; self.ndim()]).unwrap_or(0.0),
                self.iter().nth(1).unwrap_or(0.0),
                self.iter().nth(self.numel() - 2).unwrap_or(0.0),
                self.iter().last().unwrap_or(0.0),
            )
        }
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_zeros_ones() {
        let zeros = Tensor::zeros(vec![2, 3]);
        assert_eq!(zeros.sum_all(), 0.0);

        let ones = Tensor::ones(vec![2, 3]);
        assert_eq!(ones.sum_all(), 6.0);
    }

    #[test]
    fn test_slice() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let sliced = t.slice(1, 0, 2).unwrap();
        assert_eq!(sliced.shape(), &[2, 2]);
    }

    #[test]
    fn test_select() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let selected = t.select(0, 1).unwrap();
        assert_eq!(selected.shape(), &[3]);
        assert_eq!(selected.to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cat() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let c = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_tril() {
        let t = Tensor::tril(3);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(t.get(&[0, 1]).unwrap(), 0.0);
        assert_eq!(t.get(&[1, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[2, 2]).unwrap(), 1.0);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // [1,2] @ [5,6] = 1*5+2*7 = 19
        //         [7,8]   1*6+2*8 = 22
        assert_eq!(c.get(&[0, 0]).unwrap(), 19.0);
        assert_eq!(c.get(&[0, 1]).unwrap(), 22.0);
    }

    #[test]
    fn test_softmax() {
        // Use 2D tensor for realistic softmax testing
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = t.softmax(-1).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        // Each row should sum to 1
        let row0_sum: f32 = (0..3).map(|i| s.get(&[0, i]).unwrap()).sum();
        let row1_sum: f32 = (0..3).map(|i| s.get(&[1, i]).unwrap()).sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);
        assert!((row1_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]).unwrap();
        let g = t.gelu();
        // GELU(0) = 0
        assert!((g.get(&[1]).unwrap() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        let t = Tensor::from_vec(vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0], vec![2, 3]).unwrap();
        let idx = t.argmax(-1).unwrap();
        assert_eq!(idx.shape(), &[2]);
        assert_eq!(idx.get(&[0]).unwrap(), 1.0); // max at index 1 (value 3)
        assert_eq!(idx.get(&[1]).unwrap(), 2.0); // max at index 2 (value 6)
    }
}
