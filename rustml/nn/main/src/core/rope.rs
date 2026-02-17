//! Rotary Position Encoding (RoPE) and Attention with Linear Biases (ALiBi).

use crate::api::error::NnResult;
use rustml_core::{DType, Tensor, f32_vec_to_bytes};

/// Precomputed cos/sin tables for Rotary Position Encoding.
#[derive(Clone)]
pub struct RoPEFreqs {
    cos_table: Vec<f32>, // [max_seq_len, head_dim/2]
    sin_table: Vec<f32>, // [max_seq_len, head_dim/2]
    half_dim: usize,
}

impl RoPEFreqs {
    /// Build cos/sin tables for the given head_dim and maximum sequence length.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        let half_dim = head_dim / 2;
        let mut cos_table = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin_table = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_table.push(angle.cos());
                sin_table.push(angle.sin());
            }
        }

        Self { cos_table, sin_table, half_dim }
    }

    /// Build cos/sin tables with linear frequency scaling (Gemma 3 long-context).
    ///
    /// Each base frequency is divided by `scaling_factor`, effectively extending
    /// the context window by that factor. A `scaling_factor` of 1.0 is equivalent
    /// to the standard `new()` constructor.
    pub fn with_scaling(head_dim: usize, max_seq_len: usize, theta: f32, scaling_factor: f32) -> Self {
        let half_dim = head_dim / 2;
        let mut cos_table = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin_table = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = (1.0 / theta.powf(2.0 * i as f32 / head_dim as f32)) / scaling_factor;
                let angle = pos as f32 * freq;
                cos_table.push(angle.cos());
                sin_table.push(angle.sin());
            }
        }

        Self { cos_table, sin_table, half_dim }
    }

    /// Apply RoPE to tensor [B, H, S, D] starting from position `start_pos`.
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> NnResult<Tensor> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(crate::api::error::NnError::ShapeMismatch(format!(
                "RoPE expects 4D tensor [B,H,S,D], got {:?}",
                shape
            )));
        }

        let batch = shape[0];
        let heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];

        // Ensure contiguous layout for element access
        let x = if x.is_contiguous() { x.clone() } else { x.contiguous()? };
        let data = x.as_slice_f32()?;
        let mut out = Vec::with_capacity(data.len());

        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq_len {
                    let pos = start_pos + s;
                    let base = (b * heads * seq_len + h * seq_len + s) * head_dim;
                    let table_offset = pos * self.half_dim;

                    for i in 0..self.half_dim {
                        let x1 = data[base + i];
                        let x2 = data[base + self.half_dim + i];
                        let cos_val = self.cos_table[table_offset + i];
                        let sin_val = self.sin_table[table_offset + i];
                        out.push(x1 * cos_val - x2 * sin_val);
                    }
                    for i in 0..self.half_dim {
                        let x1 = data[base + i];
                        let x2 = data[base + self.half_dim + i];
                        let cos_val = self.cos_table[table_offset + i];
                        let sin_val = self.sin_table[table_offset + i];
                        out.push(x1 * sin_val + x2 * cos_val);
                    }
                }
            }
        }

        let out_bytes = f32_vec_to_bytes(out);
        Ok(Tensor::new(out_bytes, shape.to_vec(), DType::F32))
    }
}

/// Compute ALiBi slopes for `num_heads` using geometric series: 2^(-8*h/H).
pub fn compute_alibi_slopes(num_heads: usize) -> Vec<f32> {
    let mut slopes = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let exp = -8.0 * (h as f32 + 1.0) / num_heads as f32;
        slopes.push(2.0f32.powf(exp));
    }
    slopes
}

/// Build ALiBi additive bias tensor [1, H, S, T].
///
/// For causal attention, future positions get `-inf`.
/// For non-causal attention, all positions get a distance-based penalty.
pub fn alibi_bias(slopes: &[f32], seq_len: usize, total_len: usize, causal: bool) -> Tensor {
    let num_heads = slopes.len();
    let offset = total_len as isize - seq_len as isize;
    let mut data = Vec::with_capacity(num_heads * seq_len * total_len);

    for h in 0..num_heads {
        let slope = slopes[h];
        for i in 0..seq_len {
            for j in 0..total_len {
                let qi = i as isize + offset;
                let dist = j as isize - qi;
                if causal && j as isize > qi {
                    data.push(f32::NEG_INFINITY);
                } else {
                    data.push(slope * dist as f32);
                }
            }
        }
    }

    let out_bytes = f32_vec_to_bytes(data);
    Tensor::new(out_bytes, vec![1, num_heads, seq_len, total_len], DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_shape() {
        let rope = RoPEFreqs::new(64, 128, 10000.0);
        let x = Tensor::randn(vec![1, 4, 8, 64]);
        let y = rope.apply(&x, 0).unwrap();
        assert_eq!(y.shape(), &[1, 4, 8, 64]);
    }

    #[test]
    fn test_rope_with_offset() {
        let rope = RoPEFreqs::new(32, 128, 10000.0);
        let x = Tensor::randn(vec![1, 2, 1, 32]);
        let y = rope.apply(&x, 10).unwrap();
        assert_eq!(y.shape(), &[1, 2, 1, 32]);
    }

    #[test]
    fn test_rope_with_scaling() {
        let rope = RoPEFreqs::with_scaling(64, 128, 10000.0, 8.0);
        let x = Tensor::randn(vec![1, 4, 8, 64]);
        let y = rope.apply(&x, 0).unwrap();
        assert_eq!(y.shape(), &[1, 4, 8, 64]);

        // Verify scaling: frequencies should be 1/8 of unscaled
        let unscaled = RoPEFreqs::new(64, 128, 10000.0);
        let identity_scaled = RoPEFreqs::with_scaling(64, 128, 10000.0, 1.0);
        // with_scaling(factor=1.0) should match new()
        let x_small = Tensor::randn(vec![1, 1, 1, 64]);
        let y_unscaled = unscaled.apply(&x_small, 1).unwrap();
        let y_identity = identity_scaled.apply(&x_small, 1).unwrap();
        let d1 = y_unscaled.as_slice_f32().unwrap();
        let d2 = y_identity.as_slice_f32().unwrap();
        for i in 0..d1.len() {
            assert!((d1[i] - d2[i]).abs() < 1e-5, "mismatch at {}", i);
        }
    }

    #[test]
    fn test_alibi_slopes() {
        let slopes = compute_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);
        // First slope should be 2^(-1)
        assert!((slopes[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_alibi_bias_shape() {
        let slopes = compute_alibi_slopes(4);
        let bias = alibi_bias(&slopes, 8, 8, true);
        assert_eq!(bias.shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_alibi_causal_mask() {
        let slopes = compute_alibi_slopes(1);
        let bias = alibi_bias(&slopes, 3, 3, true);
        // Position (0, 1) should be -inf (future)
        assert_eq!(bias.get(&[0, 0, 0, 1]).unwrap(), f32::NEG_INFINITY);
        // Position (0, 0) should be 0 (self)
        assert_eq!(bias.get(&[0, 0, 0, 0]).unwrap(), 0.0);
    }
}
