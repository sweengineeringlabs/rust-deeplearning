use crate::error::{LLMForgeError, Result};
use super::tensor::{Tensor, f32_vec_to_bytes};
use super::dtype::DType;
use smallvec::{smallvec, SmallVec};

#[allow(non_snake_case)]
impl Tensor {
    // Matmul using faer
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        // Support broadcasting: if LHS is > 2D and RHS is 2D
        let ndim = self.shape.len();
        if ndim > 2 && other.shape.len() == 2 {
             // Collapse all dims except last one into M
             let K = self.shape[ndim - 1];
             let M: usize = self.shape[0..ndim - 1].iter().product();
             let K2 = other.shape[0];
             let N = other.shape[1];

             if K != K2 {
                  return Err(LLMForgeError::ShapeMismatch {
                    expected: vec![M, K],
                    actual: vec![other.shape[0], other.shape[1]],
                });
             }

             // Reshape LHS to [M, K]
             let lhs_2d = self.reshape(&[M, K])?;
             let out_2d = lhs_2d.matmul(other)?;

             // Reshape output back to [..., N]
             let mut out_shape: SmallVec<[usize; 4]> = SmallVec::from_slice(&self.shape[0..ndim - 1]);
             out_shape.push(N);
             return out_2d.reshape(&out_shape);
        }

        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(LLMForgeError::ShapeMismatch {
                expected: vec![2, 2],
                actual: vec![self.shape.len(), other.shape.len()],
            });
        }

        let M = self.shape[0];
        let K = self.shape[1];
        let K2 = other.shape[0];
        let N = other.shape[1];

        if K != K2 {
            return Err(LLMForgeError::ShapeMismatch {
                expected: vec![M, K],
                actual: vec![K2, N],
            });
        }

        let lhs_data = self.as_slice_f32()?;
        let rhs_data = other.as_slice_f32()?;

        let mut out_data = vec![0.0f32; M * N];

        use faer::{MatRef, MatMut};

        // Stride-aware faer matmul: C = A @ B in row-major.
        // We compute C^T = B^T @ A^T using faer's column-major convention.
        // A^T as faer [K,M]: row_stride = A.strides[1], col_stride = A.strides[0]
        // B^T as faer [N,K]: row_stride = B.strides[1], col_stride = B.strides[0]
        // This handles non-contiguous (transposed) tensors without copying.
        unsafe {
            let a_t = MatRef::<f32>::from_raw_parts(
                lhs_data.as_ptr(), K, M,
                self.strides[1] as isize, self.strides[0] as isize,
            );
            let b_t = MatRef::<f32>::from_raw_parts(
                rhs_data.as_ptr(), N, K,
                other.strides[1] as isize, other.strides[0] as isize,
            );
            let mut c_t = MatMut::<f32>::from_column_major_slice(&mut out_data, N, M);
            c_t.clone_from(b_t * a_t);
        }

        let out_bytes = f32_vec_to_bytes(out_data);

        let out_tensor = Tensor::new(out_bytes, smallvec![M, N], DType::F32);
        Ok(out_tensor)
    }

    // Batched Matrix Multiplication
    // Supports 3D [Batch, M, K] x [Batch, K, N] -> [Batch, M, N]
    // Supports 4D [Batch, Heads, M, K] x [Batch, Heads, K, N] -> [Batch, Heads, M, N]
    pub fn batched_matmul(&self, other: &Tensor) -> Result<Tensor> {
        let ndim = self.shape.len();
        if ndim != other.shape.len() || ndim < 3 {
             return Err(LLMForgeError::ShapeMismatch {
                expected: vec![3], // Minimum 3 dims for batched
                actual: vec![ndim],
            });
        }

        // Check batch dims
        let batch_dims = ndim - 2;
        for i in 0..batch_dims {
            if self.shape[i] != other.shape[i] {
                 return Err(LLMForgeError::ShapeMismatch {
                    expected: self.shape.to_vec(),
                    actual: other.shape.to_vec(),
                });
            }
        }

        let batch_count: usize = self.shape[0..batch_dims].iter().product();
        let M = self.shape[ndim - 2];
        let K = self.shape[ndim - 1];
        let K2 = other.shape[ndim - 2];
        let N = other.shape[ndim - 1];

        if K != K2 {
             return Err(LLMForgeError::ShapeMismatch {
                expected: vec![M, K],
                actual: vec![K2, N],
            });
        }

        // Output shape
        let mut out_shape: SmallVec<[usize; 4]> = SmallVec::from_slice(&self.shape[0..batch_dims]);
        out_shape.push(M);
        out_shape.push(N);

        let mut out_data = vec![0.0f32; batch_count * M * N];

        let lhs = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let rhs = if other.is_contiguous() { other.clone() } else { other.contiguous()? };

        let lhs_data = lhs.as_slice_f32()?;
        let rhs_data = rhs.as_slice_f32()?;

        use faer::{MatRef, MatMut};
        use rayon::prelude::*;

        // Parallelize over batches/heads
        out_data.par_chunks_mut(M * N)
            .zip(lhs_data.par_chunks(M * K))
            .zip(rhs_data.par_chunks(K * N))
            .for_each(|((out_chunk, lhs_chunk), rhs_chunk)| {
                let a_t = MatRef::<f32>::from_column_major_slice(lhs_chunk, K, M);
                let b_t = MatRef::<f32>::from_column_major_slice(rhs_chunk, N, K);
                let mut c_t = MatMut::<f32>::from_column_major_slice(out_chunk, N, M);

                c_t.clone_from(b_t * a_t);
            });

        let out_bytes = f32_vec_to_bytes(out_data);
        Ok(Tensor::new(out_bytes, out_shape, DType::F32))
    }

    // Element-wise addition with broadcasting support for last dim
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let lhs_data = self.as_slice_f32()?;
        let rhs_data = other.as_slice_f32()?;

        let lhs_len = lhs_data.len();
        let rhs_len = rhs_data.len();

        // Output is same shape as self (assuming self is the larger tensor or equal)
        let mut out_data = Vec::with_capacity(lhs_len);

        if lhs_len == rhs_len {
            for (a, b) in lhs_data.iter().zip(rhs_data.iter()) {
                out_data.push(a + b);
            }
        } else if rhs_len > 0 && lhs_len % rhs_len == 0 {
            // Validate broadcast shapes before proceeding
            if !is_valid_broadcast(&self.shape, &other.shape) {
                return Err(LLMForgeError::ShapeMismatch {
                    expected: self.shape.to_vec(),
                    actual: other.shape.to_vec(),
                });
            }
            // Simple broadcasting: rhs is repeated
            for (i, &a) in lhs_data.iter().enumerate() {
                out_data.push(a + rhs_data[i % rhs_len]);
            }
        } else {
             return Err(LLMForgeError::ShapeMismatch {
                expected: self.shape.to_vec(),
                actual: other.shape.to_vec(),
            });
        }

        let out_bytes = f32_vec_to_bytes(out_data);

        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        // Assume normalization over the last dimension
        if self.shape.is_empty() { return Err(LLMForgeError::ShapeMismatch{ expected: vec![1], actual: vec![]}); }
        let last_dim = self.shape[self.shape.len()-1];

        let input = self.as_slice_f32()?;
        let gamma = weight.as_slice_f32()?;
        let beta = bias.as_slice_f32()?;

        if gamma.len() != last_dim || beta.len() != last_dim {
             return Err(LLMForgeError::ShapeMismatch {
                expected: vec![last_dim],
                actual: vec![gamma.len()], // simplified
            });
        }

        let num_rows = input.len() / last_dim;
        let mut out_data = Vec::with_capacity(input.len());

        for i in 0..num_rows {
            let start = i * last_dim;
            let row = &input[start..start+last_dim];

            // Mean
            let mut sum = 0.0;
            for &x in row { sum += x; }
            let mean = sum / last_dim as f32;

            // Var
            let mut sum_sq_diff = 0.0;
            for &x in row {
                let diff = x - mean;
                sum_sq_diff += diff * diff;
            }
            let var = sum_sq_diff / last_dim as f32;
            let std = (var + eps).sqrt();

            // Normalize and Scale/Shift
            for j in 0..last_dim {
                let x = row[j];
                let norm = (x - mean) / std;
                let val = norm * gamma[j] + beta[j];
                out_data.push(val);
            }
        }

        let out_bytes = f32_vec_to_bytes(out_data);

        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    /// RMSNorm: x * weight / rms(x), where rms(x) = sqrt(mean(x^2) + eps).
    /// Used by Llama-family models instead of LayerNorm.
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor> {
        if self.shape.is_empty() {
            return Err(LLMForgeError::ShapeMismatch { expected: vec![1], actual: vec![] });
        }
        let last_dim = self.shape[self.shape.len() - 1];

        let input = self.as_slice_f32()?;
        let gamma = weight.as_slice_f32()?;

        if gamma.len() != last_dim {
            return Err(LLMForgeError::ShapeMismatch {
                expected: vec![last_dim],
                actual: vec![gamma.len()],
            });
        }

        let num_rows = input.len() / last_dim;
        let mut out_data = Vec::with_capacity(input.len());

        for i in 0..num_rows {
            let start = i * last_dim;
            let row = &input[start..start + last_dim];

            // RMS = sqrt(mean(x^2) + eps)
            let mut sum_sq = 0.0f32;
            for &x in row {
                sum_sq += x * x;
            }
            let rms = (sum_sq / last_dim as f32 + eps).sqrt();

            // Normalize and scale (no bias, no mean subtraction)
            for j in 0..last_dim {
                out_data.push(row[j] / rms * gamma[j]);
            }
        }

        let out_bytes = f32_vec_to_bytes(out_data);
        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    // GELU activation
    pub fn gelu(&self) -> Result<Tensor> {
        let input_data = self.as_slice_f32()?;
        let mut out_data = Vec::with_capacity(input_data.len());

        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();

        for &x in input_data {
            let cube = x * x * x;
            let inner = sqrt_2_pi * (x + 0.044715 * cube);
            let tanh = inner.tanh();
            let res = 0.5 * x * (1.0 + tanh);
            out_data.push(res);
        }

        let out_bytes = f32_vec_to_bytes(out_data);

        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    // SiLU activation: x * sigmoid(x)
    pub fn silu(&self) -> Result<Tensor> {
        let input_data = self.as_slice_f32()?;
        let mut out_data = Vec::with_capacity(input_data.len());

        for &x in input_data {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            out_data.push(x * sigmoid);
        }

        let out_bytes = f32_vec_to_bytes(out_data);
        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    // ReLU activation: max(0, x)
    pub fn relu(&self) -> Result<Tensor> {
        let input_data = self.as_slice_f32()?;
        let mut out_data = Vec::with_capacity(input_data.len());

        for &x in input_data {
            out_data.push(x.max(0.0));
        }

        let out_bytes = f32_vec_to_bytes(out_data);
        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        let input_data = self.as_slice_f32()?;
        let mut out_data = Vec::with_capacity(input_data.len());

        for &val in input_data {
            out_data.push(val / scalar);
        }

        let out_bytes = f32_vec_to_bytes(out_data);

        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    // Element-wise multiplication with broadcasting support
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
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
            if !is_valid_broadcast(&self.shape, &other.shape) {
                return Err(LLMForgeError::ShapeMismatch {
                    expected: self.shape.to_vec(),
                    actual: other.shape.to_vec(),
                });
            }
            for (i, &a) in lhs_data.iter().enumerate() {
                out_data.push(a * rhs_data[i % rhs_len]);
            }
        } else {
            return Err(LLMForgeError::ShapeMismatch {
                expected: self.shape.to_vec(),
                actual: other.shape.to_vec(),
            });
        }

        let out_bytes = f32_vec_to_bytes(out_data);
        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    // Element-wise cosine
    pub fn cos(&self) -> Result<Tensor> {
        let input_data = self.as_slice_f32()?;
        let mut out_data = Vec::with_capacity(input_data.len());
        for &x in input_data {
            out_data.push(x.cos());
        }
        let out_bytes = f32_vec_to_bytes(out_data);
        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    // Element-wise sine
    pub fn sin(&self) -> Result<Tensor> {
        let input_data = self.as_slice_f32()?;
        let mut out_data = Vec::with_capacity(input_data.len());
        for &x in input_data {
            out_data.push(x.sin());
        }
        let out_bytes = f32_vec_to_bytes(out_data);
        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    // Element-wise negation
    pub fn neg(&self) -> Result<Tensor> {
        let input_data = self.as_slice_f32()?;
        let mut out_data = Vec::with_capacity(input_data.len());
        for &x in input_data {
            out_data.push(-x);
        }
        let out_bytes = f32_vec_to_bytes(out_data);
        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        let input_data = self.as_slice_f32()?;
        let mut out_data = Vec::with_capacity(input_data.len());
        for &val in input_data {
            out_data.push(val * scalar);
        }
        let out_bytes = f32_vec_to_bytes(out_data);
        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }

    /// Repeat KV heads for GQA: [B, n_kv_heads, S, D] -> [B, n_kv_heads*n_rep, S, D]
    pub fn repeat_kv(&self, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(self.clone());
        }
        if self.shape.len() != 4 {
            return Err(LLMForgeError::ShapeMismatch {
                expected: vec![4],
                actual: vec![self.shape.len()],
            });
        }

        let batch = self.shape[0];
        let n_kv_heads = self.shape[1];
        let seq_len = self.shape[2];
        let head_dim = self.shape[3];

        // Ensure contiguous — input may be from transpose() with non-standard strides.
        let x = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
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

        let out_bytes = f32_vec_to_bytes(out_data);
        Ok(Tensor::new(out_bytes, smallvec![batch, out_heads, seq_len, head_dim], DType::F32))
    }

    /// Create a causal attention mask: [1, 1, seq_len, total_len]
    /// Returns 0.0 for allowed positions and NEG_INFINITY for masked positions.
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
        let out_bytes = f32_vec_to_bytes(data);
        Tensor::new(out_bytes, smallvec![1usize, 1, seq_len, total_len], DType::F32)
    }

    // Softmax along last dimension
    pub fn softmax(&self, axis: isize) -> Result<Tensor> {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };

        if axis != ndim - 1 {
             return Err(LLMForgeError::NotImplemented("Softmax only supported on last dim".into()));
        }

        let input_data = self.as_slice_f32()?;
        let last_dim_size = self.shape[ndim - 1];

        let mut out_data = vec![0.0f32; input_data.len()];

        use rayon::prelude::*;

        out_data.par_chunks_mut(last_dim_size)
            .zip(input_data.par_chunks(last_dim_size))
            .for_each(|(out_row, in_row)| {
                // Max for numerical stability
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

        let out_bytes = f32_vec_to_bytes(out_data);

        Ok(Tensor::new(out_bytes, self.shape.clone(), self.dtype))
    }
}

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
