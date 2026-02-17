//! Tensor math operations: matmul, add, softmax, activations, reductions, etc.

use crate::api::error::{TensorError, TensorResult};
use crate::api::types::DType;
use crate::core::shape::Shape;
use super::tensor::{Tensor, f32_vec_to_bytes, TensorShape};
use smallvec::{smallvec, SmallVec};

/// Check that rhs shape is a valid broadcast suffix of lhs shape.
fn is_valid_broadcast(lhs_shape: &[usize], rhs_shape: &[usize]) -> bool {
    if rhs_shape.len() > lhs_shape.len() {
        return false;
    }
    let offset = lhs_shape.len() - rhs_shape.len();
    for (i, &r) in rhs_shape.iter().enumerate() {
        if r != 1 && r != lhs_shape[offset + i] {
            return false;
        }
    }
    true
}

#[allow(non_snake_case)]
impl Tensor {
    // ==================== Element-wise binary ops ====================

    /// Element-wise addition with broadcasting.
    pub fn add(&self, other: &Tensor) -> TensorResult<Tensor> {
        let lhs_data = self.as_slice_f32()?;
        let rhs_data = other.as_slice_f32()?;
        let lhs_len = lhs_data.len();
        let rhs_len = rhs_data.len();

        let mut out_data = Vec::with_capacity(lhs_len);

        if lhs_len == rhs_len {
            for (a, b) in lhs_data.iter().zip(rhs_data.iter()) {
                out_data.push(a + b);
            }
        } else if rhs_len > 0 && lhs_len % rhs_len == 0 {
            if !is_valid_broadcast(&self.shape_sv, &other.shape_sv) {
                return Err(TensorError::BroadcastError {
                    shape1: self.shape_sv.to_vec(),
                    shape2: other.shape_sv.to_vec(),
                });
            }
            for (i, &a) in lhs_data.iter().enumerate() {
                out_data.push(a + rhs_data[i % rhs_len]);
            }
        } else {
            return Err(TensorError::BroadcastError {
                shape1: self.shape_sv.to_vec(),
                shape2: other.shape_sv.to_vec(),
            });
        }

        Ok(Tensor::new(f32_vec_to_bytes(out_data), self.shape_sv.clone(), self.dtype))
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> TensorResult<Tensor> {
        let lhs_data = self.as_slice_f32()?;
        let rhs_data = other.as_slice_f32()?;
        let lhs_len = lhs_data.len();
        let rhs_len = rhs_data.len();

        let mut out_data = Vec::with_capacity(lhs_len);

        if lhs_len == rhs_len {
            for (a, b) in lhs_data.iter().zip(rhs_data.iter()) {
                out_data.push(a - b);
            }
        } else if rhs_len > 0 && lhs_len % rhs_len == 0 {
            if !is_valid_broadcast(&self.shape_sv, &other.shape_sv) {
                return Err(TensorError::BroadcastError {
                    shape1: self.shape_sv.to_vec(),
                    shape2: other.shape_sv.to_vec(),
                });
            }
            for (i, &a) in lhs_data.iter().enumerate() {
                out_data.push(a - rhs_data[i % rhs_len]);
            }
        } else {
            return Err(TensorError::BroadcastError {
                shape1: self.shape_sv.to_vec(),
                shape2: other.shape_sv.to_vec(),
            });
        }

        Ok(Tensor::new(f32_vec_to_bytes(out_data), self.shape_sv.clone(), self.dtype))
    }

    /// Element-wise multiplication with broadcasting.
    pub fn mul(&self, other: &Tensor) -> TensorResult<Tensor> {
        let lhs_data = self.as_slice_f32()?;
        let rhs_data = other.as_slice_f32()?;
        let lhs_len = lhs_data.len();
        let rhs_len = rhs_data.len();

        let mut out_data = Vec::with_capacity(lhs_len);

        if lhs_len == rhs_len {
            for (a, b) in lhs_data.iter().zip(rhs_data.iter()) {
                out_data.push(a * b);
            }
        } else if rhs_len > 0 && lhs_len % rhs_len == 0 {
            if !is_valid_broadcast(&self.shape_sv, &other.shape_sv) {
                return Err(TensorError::BroadcastError {
                    shape1: self.shape_sv.to_vec(),
                    shape2: other.shape_sv.to_vec(),
                });
            }
            for (i, &a) in lhs_data.iter().enumerate() {
                out_data.push(a * rhs_data[i % rhs_len]);
            }
        } else {
            return Err(TensorError::BroadcastError {
                shape1: self.shape_sv.to_vec(),
                shape2: other.shape_sv.to_vec(),
            });
        }

        Ok(Tensor::new(f32_vec_to_bytes(out_data), self.shape_sv.clone(), self.dtype))
    }

    /// Element-wise division.
    pub fn div(&self, other: &Tensor) -> TensorResult<Tensor> {
        let lhs_data = self.as_slice_f32()?;
        let rhs_data = other.as_slice_f32()?;
        let lhs_len = lhs_data.len();
        let rhs_len = rhs_data.len();

        let mut out_data = Vec::with_capacity(lhs_len);

        if lhs_len == rhs_len {
            for (a, b) in lhs_data.iter().zip(rhs_data.iter()) {
                out_data.push(a / b);
            }
        } else if rhs_len > 0 && lhs_len % rhs_len == 0 {
            if !is_valid_broadcast(&self.shape_sv, &other.shape_sv) {
                return Err(TensorError::BroadcastError {
                    shape1: self.shape_sv.to_vec(),
                    shape2: other.shape_sv.to_vec(),
                });
            }
            for (i, &a) in lhs_data.iter().enumerate() {
                out_data.push(a / rhs_data[i % rhs_len]);
            }
        } else {
            return Err(TensorError::BroadcastError {
                shape1: self.shape_sv.to_vec(),
                shape2: other.shape_sv.to_vec(),
            });
        }

        Ok(Tensor::new(f32_vec_to_bytes(out_data), self.shape_sv.clone(), self.dtype))
    }

    // ==================== Scalar ops ====================

    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        self.unary_op(|x| x + scalar)
    }

    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        self.unary_op(|x| x * scalar)
    }

    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        self.unary_op(|x| x / scalar)
    }

    pub fn neg(&self) -> Tensor {
        self.unary_op(|x| -x)
    }

    pub fn sqrt(&self) -> Tensor {
        self.unary_op(|x| x.sqrt())
    }

    pub fn exp(&self) -> Tensor {
        self.unary_op(|x| x.exp())
    }

    pub fn log(&self) -> Tensor {
        self.unary_op(|x| x.ln())
    }

    pub fn pow(&self, exp: f32) -> Tensor {
        self.unary_op(|x| x.powf(exp))
    }

    pub fn abs(&self) -> Tensor {
        self.unary_op(|x| x.abs())
    }

    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        self.unary_op(|x| x.clamp(min, max))
    }

    pub fn cos(&self) -> Tensor {
        self.unary_op(|x| x.cos())
    }

    pub fn sin(&self) -> Tensor {
        self.unary_op(|x| x.sin())
    }

    pub fn tanh(&self) -> Tensor {
        self.unary_op(|x| x.tanh())
    }

    pub fn sigmoid(&self) -> Tensor {
        self.unary_op(|x| 1.0 / (1.0 + (-x).exp()))
    }

    // ==================== Activations ====================

    pub fn relu(&self) -> Tensor {
        self.unary_op(|x| x.max(0.0))
    }

    /// GELU activation (approximate).
    pub fn gelu(&self) -> Tensor {
        self.unary_op(|x| {
            let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
            0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
        })
    }

    /// SiLU (Swish) activation: x * sigmoid(x).
    pub fn silu(&self) -> Tensor {
        self.unary_op(|x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            x * sigmoid
        })
    }

    // ==================== Reductions ====================

    pub fn sum_all(&self) -> f32 {
        self.iter().sum()
    }

    pub fn mean_all(&self) -> f32 {
        self.sum_all() / self.numel() as f32
    }

    /// Sum along a dimension.
    pub fn sum(&self, dim: i64) -> TensorResult<Tensor> {
        self.reduce(dim, 0.0, |acc, x| acc + x)
    }

    /// Mean along a dimension.
    pub fn mean(&self, dim: i64) -> TensorResult<Tensor> {
        let dim_idx = self.normalize_dim(dim)?;
        let dim_size = self.shape_sv[dim_idx] as f32;
        let sum = self.sum(dim)?;
        Ok(sum.div_scalar(dim_size))
    }

    /// Variance along a dimension.
    pub fn var(&self, dim: i64) -> TensorResult<Tensor> {
        let mean = self.mean(dim)?;
        let mean_broadcast = mean.unsqueeze(dim)?.broadcast_to_shape(&self.shape_sv)?;
        let diff = self.sub(&mean_broadcast)?;
        let sq_diff = diff.mul(&diff)?;
        sq_diff.mean(dim)
    }

    /// Max along a dimension. Returns (values, indices).
    pub fn max(&self, dim: i64) -> TensorResult<(Tensor, Tensor)> {
        let dim_idx = self.normalize_dim(dim)?;
        let dim_size = self.shape_sv[dim_idx];

        let mut new_dims: Vec<usize> = self.shape_sv.to_vec();
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
            Tensor::from_vec(values, new_shape.clone())?,
            Tensor::from_vec(indices, new_shape)?,
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

        let current_dim = if depth >= reduce_dim { depth + 1 } else { depth };
        if current_dim >= self.ndim() {
            return;
        }

        for i in 0..self.shape_sv[current_dim] {
            let mut ni = current_indices.to_vec();
            ni.push(i);
            self.collect_max(values, indices, reduce_dim, dim_size, &ni, depth + 1);
        }
    }

    pub fn argmax(&self, dim: i64) -> TensorResult<Tensor> {
        let (_, indices) = self.max(dim)?;
        Ok(indices)
    }

    pub fn min(&self, dim: i64) -> TensorResult<(Tensor, Tensor)> {
        let negated = self.neg();
        let (max_vals, indices) = negated.max(dim)?;
        Ok((max_vals.neg(), indices))
    }

    // ==================== Softmax ====================

    /// Softmax along a dimension. For last-dim, uses optimized path; otherwise generic.
    pub fn softmax(&self, dim: i64) -> TensorResult<Tensor> {
        let dim_idx = self.normalize_dim(dim)?;
        let ndim = self.ndim();

        if dim_idx == ndim - 1 {
            // Fast path: softmax along last dim
            let input_data = self.as_slice_f32()?;
            let last_dim_size = self.shape_sv[ndim - 1];

            let mut out_data = vec![0.0f32; input_data.len()];

            use rayon::prelude::*;
            out_data
                .par_chunks_mut(last_dim_size)
                .zip(input_data.par_chunks(last_dim_size))
                .for_each(|(out_row, in_row)| {
                    let max_val = in_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum_exp = 0.0;
                    for (i, &val) in in_row.iter().enumerate() {
                        let exp_val = (val - max_val).exp();
                        out_row[i] = exp_val;
                        sum_exp += exp_val;
                    }
                    for val in out_row.iter_mut() {
                        *val /= sum_exp;
                    }
                });

            Ok(Tensor::new(
                f32_vec_to_bytes(out_data),
                self.shape_sv.clone(),
                self.dtype,
            ))
        } else {
            // Generic path via reductions
            let max_vals = self.max(dim)?.0;
            let max_broadcast = max_vals.unsqueeze(dim)?;
            let max_broadcast = max_broadcast.broadcast_to_shape(&self.shape_sv)?;
            let shifted = self.sub(&max_broadcast)?;
            let exp_vals = shifted.exp();
            let sum_exp = exp_vals.sum(dim)?;
            let sum_broadcast = sum_exp.unsqueeze(dim)?;
            let sum_broadcast = sum_broadcast.broadcast_to_shape(&self.shape_sv)?;
            exp_vals.div(&sum_broadcast)
        }
    }

    // ==================== Layer normalization ====================

    /// LayerNorm over the last dimension.
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> TensorResult<Tensor> {
        if self.shape_sv.is_empty() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![1],
                got: vec![],
            });
        }
        let last_dim = self.shape_sv[self.shape_sv.len() - 1];

        let input = self.as_slice_f32()?;
        let gamma = weight.as_slice_f32()?;
        let beta = bias.as_slice_f32()?;

        if gamma.len() != last_dim || beta.len() != last_dim {
            return Err(TensorError::ShapeMismatch {
                expected: vec![last_dim],
                got: vec![gamma.len()],
            });
        }

        let num_rows = input.len() / last_dim;
        let mut out_data = Vec::with_capacity(input.len());

        for i in 0..num_rows {
            let start = i * last_dim;
            let row = &input[start..start + last_dim];

            let mut sum = 0.0;
            for &x in row {
                sum += x;
            }
            let mean = sum / last_dim as f32;

            let mut sum_sq_diff = 0.0;
            for &x in row {
                let diff = x - mean;
                sum_sq_diff += diff * diff;
            }
            let var = sum_sq_diff / last_dim as f32;
            let std = (var + eps).sqrt();

            for j in 0..last_dim {
                let norm = (row[j] - mean) / std;
                out_data.push(norm * gamma[j] + beta[j]);
            }
        }

        Ok(Tensor::new(
            f32_vec_to_bytes(out_data),
            self.shape_sv.clone(),
            self.dtype,
        ))
    }

    /// RMSNorm over the last dimension.
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> TensorResult<Tensor> {
        if self.shape_sv.is_empty() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![1],
                got: vec![],
            });
        }
        let last_dim = self.shape_sv[self.shape_sv.len() - 1];
        let input = self.as_slice_f32()?;
        let gamma = weight.as_slice_f32()?;

        if gamma.len() != last_dim {
            return Err(TensorError::ShapeMismatch {
                expected: vec![last_dim],
                got: vec![gamma.len()],
            });
        }

        let num_rows = input.len() / last_dim;
        let mut out_data = Vec::with_capacity(input.len());

        for i in 0..num_rows {
            let start = i * last_dim;
            let row = &input[start..start + last_dim];

            let mut sum_sq = 0.0f32;
            for &x in row {
                sum_sq += x * x;
            }
            let rms = (sum_sq / last_dim as f32 + eps).sqrt();

            for j in 0..last_dim {
                out_data.push(row[j] / rms * gamma[j]);
            }
        }

        Ok(Tensor::new(
            f32_vec_to_bytes(out_data),
            self.shape_sv.clone(),
            self.dtype,
        ))
    }

    // ==================== Matrix multiplication ====================

    /// Matrix multiplication using faer for 2D, with broadcasting for higher dims.
    pub fn matmul(&self, other: &Tensor) -> TensorResult<Tensor> {
        let ndim = self.shape_sv.len();
        let other_ndim = other.shape_sv.len();

        // Broadcasting: if LHS is >2D and RHS is 2D, collapse batch dims
        if ndim > 2 && other_ndim == 2 {
            let K = self.shape_sv[ndim - 1];
            let M: usize = self.shape_sv[0..ndim - 1].iter().product();
            let K2 = other.shape_sv[0];
            let N = other.shape_sv[1];

            if K != K2 {
                return Err(TensorError::MatmulDimensionMismatch { left: K, right: K2 });
            }

            let lhs_2d = self.reshape(&[M, K])?;
            let out_2d = lhs_2d.matmul(other)?;

            let mut out_shape: SmallVec<[usize; 4]> =
                SmallVec::from_slice(&self.shape_sv[0..ndim - 1]);
            out_shape.push(N);
            return out_2d.reshape(&out_shape);
        }

        // Same-dim batched matmul (e.g. 4D x 4D for attention)
        if ndim == other_ndim && ndim > 2 {
            return self.batched_matmul(other);
        }

        if ndim != 2 || other_ndim != 2 {
            return Err(TensorError::InvalidOperation(
                "Matrix multiplication requires 2D tensors (or broadcasting >2D x 2D, or same-dim batched)".into(),
            ));
        }

        let M = self.shape_sv[0];
        let K = self.shape_sv[1];
        let K2 = other.shape_sv[0];
        let N = other.shape_sv[1];

        if K != K2 {
            return Err(TensorError::MatmulDimensionMismatch { left: K, right: K2 });
        }

        let lhs_data = self.as_slice_f32()?;
        let rhs_data = other.as_slice_f32()?;

        let mut out_data = vec![0.0f32; M * N];

        // C^T = B^T @ A^T using faer column-major convention
        unsafe {
            let a_t = faer::mat::from_raw_parts::<f32, usize, usize>(
                lhs_data.as_ptr(),
                K,
                M,
                self.strides[1] as isize,
                self.strides[0] as isize,
            );
            let b_t = faer::mat::from_raw_parts::<f32, usize, usize>(
                rhs_data.as_ptr(),
                N,
                K,
                other.strides[1] as isize,
                other.strides[0] as isize,
            );
            let mut c_t = faer::mat::from_column_major_slice_mut(&mut out_data, N, M);
            c_t.copy_from(b_t * a_t);
        }

        Ok(Tensor::new(
            f32_vec_to_bytes(out_data),
            smallvec![M, N],
            DType::F32,
        ))
    }

    /// Batched matrix multiplication: [B, M, K] x [B, K, N] -> [B, M, N].
    pub fn batched_matmul(&self, other: &Tensor) -> TensorResult<Tensor> {
        let ndim = self.shape_sv.len();
        if ndim != other.shape_sv.len() || ndim < 3 {
            return Err(TensorError::InvalidOperation(
                "batched_matmul requires >=3D tensors of same ndim".into(),
            ));
        }

        let batch_dims = ndim - 2;
        for i in 0..batch_dims {
            if self.shape_sv[i] != other.shape_sv[i] {
                return Err(TensorError::ShapeMismatch {
                    expected: self.shape_sv.to_vec(),
                    got: other.shape_sv.to_vec(),
                });
            }
        }

        let batch_count: usize = self.shape_sv[0..batch_dims].iter().product();
        let M = self.shape_sv[ndim - 2];
        let K = self.shape_sv[ndim - 1];
        let K2 = other.shape_sv[ndim - 2];
        let N = other.shape_sv[ndim - 1];

        if K != K2 {
            return Err(TensorError::MatmulDimensionMismatch { left: K, right: K2 });
        }

        let mut out_shape: SmallVec<[usize; 4]> =
            SmallVec::from_slice(&self.shape_sv[0..batch_dims]);
        out_shape.push(M);
        out_shape.push(N);

        let mut out_data = vec![0.0f32; batch_count * M * N];

        let lhs = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()?
        };
        let rhs = if other.is_contiguous() {
            other.clone()
        } else {
            other.contiguous()?
        };

        let lhs_data = lhs.as_slice_f32()?;
        let rhs_data = rhs.as_slice_f32()?;

        use rayon::prelude::*;

        out_data
            .par_chunks_mut(M * N)
            .zip(lhs_data.par_chunks(M * K))
            .zip(rhs_data.par_chunks(K * N))
            .for_each(|((out_chunk, lhs_chunk), rhs_chunk)| {
                let a_t = faer::mat::from_column_major_slice(lhs_chunk, K, M);
                let b_t = faer::mat::from_column_major_slice(rhs_chunk, N, K);
                let mut c_t = faer::mat::from_column_major_slice_mut(out_chunk, N, M);
                c_t.copy_from(b_t * a_t);
            });

        Ok(Tensor::new(
            f32_vec_to_bytes(out_data),
            out_shape,
            DType::F32,
        ))
    }

    // ==================== Causal mask ====================

    /// Create a causal attention mask: [1, 1, seq_len, total_len].
    pub fn causal_mask(seq_len: usize, total_len: usize) -> Tensor {
        let mut data = Vec::with_capacity(seq_len * total_len);
        let offset = total_len - seq_len;
        for i in 0..seq_len {
            for j in 0..total_len {
                if j <= i + offset {
                    data.push(0.0f32);
                } else {
                    data.push(f32::NEG_INFINITY);
                }
            }
        }
        Tensor::new(
            f32_vec_to_bytes(data),
            smallvec![1usize, 1, seq_len, total_len],
            DType::F32,
        )
    }

    /// Create a sliding-window causal mask: [1, 1, seq_len, total_len].
    ///
    /// Position `i` (query) attends to position `j` (key) when:
    ///   - `j <= query_pos` (causal: no future tokens)
    ///   - `j >= query_pos - window_size + 1` (window: only recent tokens)
    ///
    /// where `query_pos = i + offset` and `offset = total_len - seq_len`.
    /// When `window_size >= total_len`, this equals `causal_mask()`.
    pub fn sliding_window_mask(seq_len: usize, total_len: usize, window_size: usize) -> Tensor {
        let mut data = Vec::with_capacity(seq_len * total_len);
        let offset = total_len - seq_len;
        for i in 0..seq_len {
            let query_pos = i + offset;
            for j in 0..total_len {
                let causal = j <= query_pos;
                let in_window = (query_pos as isize - j as isize) < window_size as isize;
                if causal && in_window {
                    data.push(0.0f32);
                } else {
                    data.push(f32::NEG_INFINITY);
                }
            }
        }
        Tensor::new(
            f32_vec_to_bytes(data),
            smallvec![1usize, 1, seq_len, total_len],
            DType::F32,
        )
    }

    /// Repeat KV heads for GQA: [B, n_kv_heads, S, D] -> [B, n_kv_heads*n_rep, S, D].
    pub fn repeat_kv(&self, n_rep: usize) -> TensorResult<Tensor> {
        if n_rep == 1 {
            return Ok(self.clone());
        }
        if self.shape_sv.len() != 4 {
            return Err(TensorError::InvalidOperation(
                "repeat_kv requires 4D tensor".into(),
            ));
        }

        let batch = self.shape_sv[0];
        let n_kv_heads = self.shape_sv[1];
        let seq_len = self.shape_sv[2];
        let head_dim = self.shape_sv[3];

        let x = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()?
        };
        let input_data = x.as_slice_f32()?;
        let out_heads = n_kv_heads * n_rep;
        let mut out_data = Vec::with_capacity(batch * out_heads * seq_len * head_dim);

        let head_size = seq_len * head_dim;
        for b in 0..batch {
            for h in 0..n_kv_heads {
                let start = (b * n_kv_heads + h) * head_size;
                let head_data = &input_data[start..start + head_size];
                for _ in 0..n_rep {
                    out_data.extend_from_slice(head_data);
                }
            }
        }

        Ok(Tensor::new(
            f32_vec_to_bytes(out_data),
            smallvec![batch, out_heads, seq_len, head_dim],
            DType::F32,
        ))
    }

    // ==================== Masked fill ====================

    /// Fill tensor where mask is true with specified value.
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> TensorResult<Tensor> {
        let broadcast_shape = Shape::new(self.shape_sv.to_vec())
            .broadcast_with(&Shape::new(mask.shape_sv.to_vec()))
            .ok_or_else(|| TensorError::BroadcastError {
                shape1: self.shape_sv.to_vec(),
                shape2: mask.shape_sv.to_vec(),
            })?;

        let self_broadcast = self.broadcast_to(&broadcast_shape)?;
        let mask_broadcast = mask.broadcast_to(&broadcast_shape)?;

        let new_data: Vec<f32> = self_broadcast
            .iter()
            .zip(mask_broadcast.iter())
            .map(|(v, m)| if m != 0.0 { value } else { v })
            .collect();

        Tensor::from_vec(new_data, broadcast_shape)
    }

    // ==================== Concatenation ====================

    /// Concatenate tensors along a dimension.
    pub fn cat(tensors: &[&Tensor], dim: i64) -> TensorResult<Tensor> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyTensor);
        }

        let first = tensors[0];
        let dim_idx = first.normalize_dim(dim)?;

        for t in tensors.iter().skip(1) {
            if t.ndim() != first.ndim() {
                return Err(TensorError::ShapeMismatch {
                    expected: first.shape_sv.to_vec(),
                    got: t.shape_sv.to_vec(),
                });
            }
            for (i, (&s1, &s2)) in first.shape_sv.iter().zip(t.shape_sv.iter()).enumerate() {
                if i != dim_idx && s1 != s2 {
                    return Err(TensorError::ShapeMismatch {
                        expected: first.shape_sv.to_vec(),
                        got: t.shape_sv.to_vec(),
                    });
                }
            }
        }

        let total_dim_size: usize = tensors.iter().map(|t| t.shape_sv[dim_idx]).sum();
        let mut new_dims = first.shape_sv.to_vec();
        new_dims[dim_idx] = total_dim_size;
        let new_shape = Shape::new(new_dims);

        let mut new_data = Vec::with_capacity(new_shape.numel());
        Self::collect_cat(&mut new_data, tensors, dim_idx, &[], 0);

        Tensor::from_vec(new_data, new_shape)
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
            let cat_idx = indices[cat_dim];
            let mut offset = 0;
            for t in tensors {
                let dim_size = t.shape_sv[cat_dim];
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
            let total: usize = tensors.iter().map(|t| t.shape_sv[cat_dim]).sum();
            0..total
        } else {
            0..tensors[0].shape_sv[depth]
        };

        for i in range {
            let mut new_indices = indices.to_vec();
            new_indices.push(i);
            Self::collect_cat(result, tensors, cat_dim, &new_indices, depth + 1);
        }
    }

    // ==================== Internal helpers ====================

    fn unary_op(&self, f: impl Fn(f32) -> f32) -> Tensor {
        let data: Vec<f32> = self.iter().map(f).collect();
        Tensor::from_vec(data, self.shape_sv.to_vec()).unwrap()
    }

    fn reduce(&self, dim: i64, init: f32, f: impl Fn(f32, f32) -> f32) -> TensorResult<Tensor> {
        let dim_idx = self.normalize_dim(dim)?;

        let mut new_dims = self.shape_sv.to_vec();
        new_dims.remove(dim_idx);
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(new_dims)
        };

        let dim_size = self.shape_sv[dim_idx];
        let mut new_data = Vec::with_capacity(new_shape.numel());

        self.collect_reduce(&mut new_data, dim_idx, dim_size, init, &f, &[], 0);

        Tensor::from_vec(new_data, new_shape)
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

        for i in 0..self.shape_sv[current_dim] {
            let mut ni = indices.to_vec();
            ni.push(i);
            self.collect_reduce(result, reduce_dim, dim_size, init, f, &ni, depth + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get(&[0, 0]).unwrap(), 19.0);
        assert_eq!(c.get(&[0, 1]).unwrap(), 22.0);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = t.softmax(-1).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        let row0_sum: f32 = (0..3).map(|i| s.get(&[0, i]).unwrap()).sum();
        let row1_sum: f32 = (0..3).map(|i| s.get(&[1, i]).unwrap()).sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);
        assert!((row1_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]).unwrap();
        let g = t.gelu();
        assert!((g.get(&[1]).unwrap() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        let t = Tensor::from_vec(vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0], vec![2, 3]).unwrap();
        let idx = t.argmax(-1).unwrap();
        assert_eq!(idx.shape(), &[2]);
        assert_eq!(idx.get(&[0]).unwrap(), 1.0);
        assert_eq!(idx.get(&[1]).unwrap(), 2.0);
    }

    #[test]
    fn test_add_broadcast() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(c.get(&[1, 2]).unwrap(), 36.0);
    }

    #[test]
    fn test_layer_norm() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let w = Tensor::ones(vec![3]);
        let b = Tensor::zeros(vec![3]);
        let ln = t.layer_norm(&w, &b, 1e-5).unwrap();
        assert_eq!(ln.shape(), &[2, 3]);
        // After layer norm, each row should have mean ≈ 0
        let mean0: f32 = (0..3).map(|i| ln.get(&[0, i]).unwrap()).sum::<f32>() / 3.0;
        assert!(mean0.abs() < 1e-5);
    }

    #[test]
    fn test_sliding_window_mask_shape() {
        let mask = Tensor::sliding_window_mask(4, 4, 2);
        assert_eq!(mask.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_sliding_window_mask_values() {
        // seq_len=4, total_len=4, window=2
        // Row 0 (query_pos=0): attends to j=0 only (causal, window=[0,0])
        // Row 1 (query_pos=1): attends to j=0,1 (causal, window=[0,1])
        // Row 2 (query_pos=2): attends to j=1,2 (causal, window=[1,2])
        // Row 3 (query_pos=3): attends to j=2,3 (causal, window=[2,3])
        let mask = Tensor::sliding_window_mask(4, 4, 2);
        // Row 0: [0, -inf, -inf, -inf]
        assert_eq!(mask.get(&[0, 0, 0, 0]).unwrap(), 0.0);
        assert_eq!(mask.get(&[0, 0, 0, 1]).unwrap(), f32::NEG_INFINITY);
        // Row 1: [0, 0, -inf, -inf]
        assert_eq!(mask.get(&[0, 0, 1, 0]).unwrap(), 0.0);
        assert_eq!(mask.get(&[0, 0, 1, 1]).unwrap(), 0.0);
        assert_eq!(mask.get(&[0, 0, 1, 2]).unwrap(), f32::NEG_INFINITY);
        // Row 2: [-inf, 0, 0, -inf]  (window excludes j=0)
        assert_eq!(mask.get(&[0, 0, 2, 0]).unwrap(), f32::NEG_INFINITY);
        assert_eq!(mask.get(&[0, 0, 2, 1]).unwrap(), 0.0);
        assert_eq!(mask.get(&[0, 0, 2, 2]).unwrap(), 0.0);
        assert_eq!(mask.get(&[0, 0, 2, 3]).unwrap(), f32::NEG_INFINITY);
        // Row 3: [-inf, -inf, 0, 0]
        assert_eq!(mask.get(&[0, 0, 3, 1]).unwrap(), f32::NEG_INFINITY);
        assert_eq!(mask.get(&[0, 0, 3, 2]).unwrap(), 0.0);
        assert_eq!(mask.get(&[0, 0, 3, 3]).unwrap(), 0.0);
    }

    #[test]
    fn test_sliding_window_large_window_equals_causal() {
        let seq_len = 5;
        let causal = Tensor::causal_mask(seq_len, seq_len);
        let sliding = Tensor::sliding_window_mask(seq_len, seq_len, seq_len);
        let causal_data = causal.as_slice_f32().unwrap();
        let sliding_data = sliding.as_slice_f32().unwrap();
        for i in 0..causal_data.len() {
            assert_eq!(causal_data[i], sliding_data[i],
                "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_sliding_window_decode_step_with_offset() {
        // Simulates a decode step: seq_len=1, total_len=5, window=3
        // query_pos = 0 + (5-1) = 4, so attends to j in [2,4]
        let mask = Tensor::sliding_window_mask(1, 5, 3);
        assert_eq!(mask.shape(), &[1, 1, 1, 5]);
        assert_eq!(mask.get(&[0, 0, 0, 0]).unwrap(), f32::NEG_INFINITY);
        assert_eq!(mask.get(&[0, 0, 0, 1]).unwrap(), f32::NEG_INFINITY);
        assert_eq!(mask.get(&[0, 0, 0, 2]).unwrap(), 0.0);
        assert_eq!(mask.get(&[0, 0, 0, 3]).unwrap(), 0.0);
        assert_eq!(mask.get(&[0, 0, 0, 4]).unwrap(), 0.0);
    }
}
