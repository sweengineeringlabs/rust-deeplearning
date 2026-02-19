//! Rotary Position Encoding (RoPE) and Attention with Linear Biases (ALiBi).

use std::time::Instant;
use crate::api::error::NnResult;
use rustml_core::{DType, Tensor, f32_vec_to_bytes};

/// Apply RoPE rotation to half_dim pairs using AVX2.
/// Writes both halves of head_dim into contiguous out slice.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rope_apply_avx2(
    x1: &[f32],       // [half_dim] — first half of head
    x2: &[f32],       // [half_dim] — second half of head
    cos: &[f32],      // [half_dim]
    sin: &[f32],      // [half_dim]
    out: &mut [f32],   // [head_dim] — first half then second half
    half_dim: usize,
) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= half_dim {
        let v_x1 = _mm256_loadu_ps(x1.as_ptr().add(i));
        let v_x2 = _mm256_loadu_ps(x2.as_ptr().add(i));
        let v_cos = _mm256_loadu_ps(cos.as_ptr().add(i));
        let v_sin = _mm256_loadu_ps(sin.as_ptr().add(i));

        // first_half[i] = x1*cos - x2*sin
        let first = _mm256_sub_ps(
            _mm256_mul_ps(v_x1, v_cos),
            _mm256_mul_ps(v_x2, v_sin),
        );
        // second_half[i] = x1*sin + x2*cos
        let second = _mm256_add_ps(
            _mm256_mul_ps(v_x1, v_sin),
            _mm256_mul_ps(v_x2, v_cos),
        );

        _mm256_storeu_ps(out.as_mut_ptr().add(i), first);
        _mm256_storeu_ps(out.as_mut_ptr().add(half_dim + i), second);
        i += 8;
    }
    // Scalar remainder
    while i < half_dim {
        out[i] = x1[i] * cos[i] - x2[i] * sin[i];
        out[half_dim + i] = x1[i] * sin[i] + x2[i] * cos[i];
        i += 1;
    }
}

/// Apply RoPE rotation to half_dim pairs using NEON.
#[cfg(target_arch = "aarch64")]
unsafe fn rope_apply_neon(
    x1: &[f32],
    x2: &[f32],
    cos: &[f32],
    sin: &[f32],
    out: &mut [f32],
    half_dim: usize,
) {
    use std::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= half_dim {
        let v_x1 = vld1q_f32(x1.as_ptr().add(i));
        let v_x2 = vld1q_f32(x2.as_ptr().add(i));
        let v_cos = vld1q_f32(cos.as_ptr().add(i));
        let v_sin = vld1q_f32(sin.as_ptr().add(i));

        // first = x1*cos - x2*sin
        let first = vsubq_f32(vmulq_f32(v_x1, v_cos), vmulq_f32(v_x2, v_sin));
        // second = x1*sin + x2*cos
        let second = vaddq_f32(vmulq_f32(v_x1, v_sin), vmulq_f32(v_x2, v_cos));

        vst1q_f32(out.as_mut_ptr().add(i), first);
        vst1q_f32(out.as_mut_ptr().add(half_dim + i), second);
        i += 4;
    }
    while i < half_dim {
        out[i] = x1[i] * cos[i] - x2[i] * sin[i];
        out[half_dim + i] = x1[i] * sin[i] + x2[i] * cos[i];
        i += 1;
    }
}

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
    /// SIMD-accelerated on x86_64 (AVX2) and aarch64 (NEON).
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> NnResult<Tensor> {
        let _t = if log::log_enabled!(log::Level::Trace) { Some(Instant::now()) } else { None };
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
        let mut out = vec![0.0f32; data.len()];

        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2") && self.half_dim >= 8;
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq_len {
                    let pos = start_pos + s;
                    let base = (b * heads * seq_len + h * seq_len + s) * head_dim;
                    let table_offset = pos * self.half_dim;

                    let x1 = &data[base..base + self.half_dim];
                    let x2 = &data[base + self.half_dim..base + head_dim];
                    let cos_slice = &self.cos_table[table_offset..table_offset + self.half_dim];
                    let sin_slice = &self.sin_table[table_offset..table_offset + self.half_dim];
                    let out_slice = &mut out[base..base + head_dim];

                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        unsafe {
                            rope_apply_avx2(x1, x2, cos_slice, sin_slice, out_slice, self.half_dim);
                        }
                        continue;
                    }

                    #[cfg(target_arch = "aarch64")]
                    if self.half_dim >= 4 {
                        unsafe {
                            rope_apply_neon(x1, x2, cos_slice, sin_slice, out_slice, self.half_dim);
                        }
                        continue;
                    }

                    // Scalar fallback
                    for i in 0..self.half_dim {
                        out_slice[i] = x1[i] * cos_slice[i] - x2[i] * sin_slice[i];
                        out_slice[self.half_dim + i] = x1[i] * sin_slice[i] + x2[i] * cos_slice[i];
                    }
                }
            }
        }

        let out_bytes = f32_vec_to_bytes(out);
        let result = Tensor::new(out_bytes, shape.to_vec(), DType::F32);
        if let Some(t) = _t {
            log::trace!("[perf] rope::apply {:?} pos={} {:.3}ms", shape, start_pos, t.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(result)
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
