pub mod cross;
pub use cross::CrossAttention;

use crate::config::PositionEncoding;
use crate::core::tensor::{Tensor, DType, f32_vec_to_bytes};
use crate::error::{LLMForgeError, Result};
use crate::nn::{Linear, Layer};

/// Precomputed cos/sin tables for RoPE.
pub struct RoPEFreqs {
    cos_table: Vec<f32>, // [max_seq_len, head_dim/2]
    sin_table: Vec<f32>, // [max_seq_len, head_dim/2]
    half_dim: usize,
}

impl RoPEFreqs {
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

    /// Apply RoPE to tensor [B, H, S, D] starting from position start_pos.
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(LLMForgeError::ShapeMismatch {
                expected: vec![4],
                actual: vec![shape.len()],
            });
        }
        let batch = shape[0];
        let heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];

        // Ensure contiguous layout — after reshape().transpose() the tensor
        // has [B,H,S,D] shape but [B,S,H,D] storage order.
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

/// Compute ALiBi slopes for num_heads using geometric series: 2^(-8*h/H).
pub fn compute_alibi_slopes(num_heads: usize) -> Vec<f32> {
    let mut slopes = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let exp = -8.0 * (h as f32 + 1.0) / num_heads as f32;
        slopes.push(2.0f32.powf(exp));
    }
    slopes
}

/// Build ALiBi additive bias tensor [1, H, S, T].
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
                    // ALiBi: slope * (j - query_position) where dist <= 0 for past tokens
                    data.push(slope * dist as f32);
                }
            }
        }
    }

    let out_bytes = f32_vec_to_bytes(data);
    Tensor::new(out_bytes, vec![1, num_heads, seq_len, total_len], DType::F32)
}

pub struct MultiHeadAttention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    d_model: usize,
    causal: bool,
    position_encoding: PositionEncoding,
    rope_freqs: Option<RoPEFreqs>,
    alibi_slopes: Option<Vec<f32>>,

    // Projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        bias: bool,
        causal: bool,
        position_encoding: PositionEncoding,
        max_seq_len: usize,
        rope_theta: f32,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(LLMForgeError::InvalidConfig(
                format!("d_model ({}) must be divisible by num_heads ({})", d_model, num_heads)
            ));
        }
        let head_dim = d_model / num_heads;
        let n_kv = num_kv_heads.unwrap_or(num_heads);

        if num_heads % n_kv != 0 {
            return Err(LLMForgeError::InvalidConfig(
                format!("num_heads ({}) must be divisible by num_kv_heads ({})", num_heads, n_kv)
            ));
        }

        let kv_dim = n_kv * head_dim;

        let rope_freqs = if position_encoding == PositionEncoding::RoPE {
            Some(RoPEFreqs::new(head_dim, max_seq_len, rope_theta))
        } else {
            None
        };

        let alibi_slopes = if position_encoding == PositionEncoding::ALiBi {
            Some(compute_alibi_slopes(num_heads))
        } else {
            None
        };

        Ok(Self {
            num_heads,
            num_kv_heads: n_kv,
            head_dim,
            d_model,
            causal,
            position_encoding,
            rope_freqs,
            alibi_slopes,
            q_proj: Linear::new(d_model, d_model, bias),
            k_proj: Linear::new(d_model, kv_dim, bias),
            v_proj: Linear::new(d_model, kv_dim, bias),
            out_proj: Linear::new(d_model, d_model, bias),
        })
    }

    /// Construct from pre-loaded projection layers.
    pub fn from_weights(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        causal: bool,
        position_encoding: PositionEncoding,
        max_seq_len: usize,
        rope_theta: f32,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(LLMForgeError::InvalidConfig(
                format!("d_model ({}) must be divisible by num_heads ({})", d_model, num_heads)
            ));
        }
        let head_dim = d_model / num_heads;
        let n_kv = num_kv_heads.unwrap_or(num_heads);

        let rope_freqs = if position_encoding == PositionEncoding::RoPE {
            Some(RoPEFreqs::new(head_dim, max_seq_len, rope_theta))
        } else {
            None
        };

        let alibi_slopes = if position_encoding == PositionEncoding::ALiBi {
            Some(compute_alibi_slopes(num_heads))
        } else {
            None
        };

        Ok(Self {
            num_heads,
            num_kv_heads: n_kv,
            head_dim,
            d_model,
            causal,
            position_encoding,
            rope_freqs,
            alibi_slopes,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    /// Number of KV heads (for cache sizing).
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Toggle native Q4_0×Q8_0 integer matmul on all Linear layers.
    pub fn set_native_q4_matmul(&mut self, enabled: bool) {
        self.q_proj.use_native_q4 = enabled;
        self.k_proj.use_native_q4 = enabled;
        self.v_proj.use_native_q4 = enabled;
        self.out_proj.use_native_q4 = enabled;
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let (mut total, mut frozen) = (0, 0);
        for proj in [&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj] {
            let (t, f) = proj.parameter_count();
            total += t;
            frozen += f;
        }
        (total, frozen)
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Project inputs
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // Reshape Q: [B, S, num_heads, D] -> [B, num_heads, S, D]
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?;

        // Reshape K/V: [B, S, num_kv_heads, D] -> [B, num_kv_heads, S, D]
        let k = k.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
                 .transpose(1, 2)?;
        let v = v.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
                 .transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = if let Some(ref rope) = self.rope_freqs {
            (rope.apply(&q, 0)?, rope.apply(&k, 0)?)
        } else {
            (q, k)
        };

        // GQA: expand K/V to match Q heads
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = k.repeat_kv(n_rep)?;
        let v = v.repeat_kv(n_rep)?;

        // Attention scores
        let k_t = k.transpose(2, 3)?;
        let scores = q.batched_matmul(&k_t)?;
        let scale = (self.head_dim as f32).sqrt();
        let scores = scores.div_scalar(scale)?;

        // Apply mask/bias
        let scores = if let Some(ref slopes) = self.alibi_slopes {
            let bias = alibi_bias(slopes, seq_len, seq_len, self.causal);
            scores.add(&bias)?
        } else if self.causal && seq_len > 1 {
            let mask = Tensor::causal_mask(seq_len, seq_len);
            scores.add(&mask)?
        } else {
            scores
        };

        let attn = scores.softmax(-1)?;
        let context = attn.batched_matmul(&v)?;

        let context = context.transpose(1, 2)?
                             .reshape(&[batch_size, seq_len, self.d_model])?;

        self.out_proj.forward(&context)
    }

    pub fn forward_with_cache(&self, input: &Tensor, cache: &mut KVCache, layer_idx: usize) -> Result<Tensor> {
        if cache.head_dim() != self.head_dim {
            return Err(LLMForgeError::InvalidConfig(
                format!("KVCache head_dim ({}) does not match attention head_dim ({})", cache.head_dim(), self.head_dim)
            ));
        }

        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let start_pos = cache.current_len;

        // Project
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // Reshape Q: [B, S, num_heads, D] -> [B, num_heads, S, D]
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?;

        // Reshape K/V: [B, S, num_kv_heads, D] -> [B, num_kv_heads, S, D]
        let k = k.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
                 .transpose(1, 2)?;
        let v = v.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
                 .transpose(1, 2)?;

        // Apply RoPE before cache update
        let (q, k) = if let Some(ref rope) = self.rope_freqs {
            (rope.apply(&q, start_pos)?, rope.apply(&k, start_pos)?)
        } else {
            (q, k)
        };

        // Update cache
        cache.update(layer_idx, k.clone(), v.clone())?;

        // Get full K/V history
        let total_len = cache.current_len + seq_len;
        let (k_full, v_full) = cache.get_view(layer_idx, total_len)?;

        // GQA: expand K/V
        let n_rep = self.num_heads / self.num_kv_heads;
        let k_full = k_full.repeat_kv(n_rep)?;
        let v_full = v_full.repeat_kv(n_rep)?;

        // Attention scores
        let k_t = k_full.transpose(2, 3)?;
        let scores = q.batched_matmul(&k_t)?;
        let scale = (self.head_dim as f32).sqrt();
        let scores = scores.div_scalar(scale)?;

        // Apply mask/bias
        let scores = if let Some(ref slopes) = self.alibi_slopes {
            let bias = alibi_bias(slopes, seq_len, total_len, self.causal);
            scores.add(&bias)?
        } else if self.causal && seq_len > 1 {
            let mask = Tensor::causal_mask(seq_len, total_len);
            scores.add(&mask)?
        } else {
            scores
        };

        let attn = scores.softmax(-1)?;
        let context = attn.batched_matmul(&v_full)?;

        let context = context.transpose(1, 2)?
                             .reshape(&[batch_size, seq_len, self.d_model])?;

        self.out_proj.forward(&context)
    }
}

pub struct KVCache {
    past_keys: Vec<Tensor>,
    past_values: Vec<Tensor>,
    max_seq_len: usize,
    pub current_len: usize,
    head_dim: usize,
    num_kv_heads: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, max_seq_len: usize, head_dim: usize, num_kv_heads: usize) -> Self {
        let key_shape = vec![1, num_kv_heads, max_seq_len, head_dim];
        let val_shape = vec![1, num_kv_heads, max_seq_len, head_dim];

        let past_keys = (0..num_layers)
            .map(|_| Tensor::zeros(&key_shape))
            .collect();

        let past_values = (0..num_layers)
            .map(|_| Tensor::zeros(&val_shape))
            .collect();

        Self {
            past_keys,
            past_values,
            max_seq_len,
            current_len: 0,
            head_dim,
            num_kv_heads,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    // Returns view of `0..len`
    pub fn get_view(&self, layer_idx: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let k = self.past_keys[layer_idx].slice_sequence(0, len)?;
        let v = self.past_values[layer_idx].slice_sequence(0, len)?;
        Ok((k, v))
    }

    pub fn update(&mut self, layer_idx: usize, key: Tensor, value: Tensor) -> Result<()> {
        let seq_len = key.shape()[2];
        if self.current_len + seq_len > self.max_seq_len {
            return Err(LLMForgeError::SequenceLengthExceeded {
                max: self.max_seq_len,
                actual: self.current_len + seq_len,
            });
        }
        self.past_keys[layer_idx].slice_assign_sequence(self.current_len, &key)?;
        self.past_values[layer_idx].slice_assign_sequence(self.current_len, &value)?;
        Ok(())
    }

    pub fn advance(&mut self, step: usize) {
        self.current_len += step;
    }

    /// Create a deep copy with independently owned tensor data (not Arc-shared).
    /// Required for beam search where each beam needs a mutable cache.
    pub fn deep_clone(&self) -> Result<Self> {
        let past_keys = self.past_keys.iter()
            .map(|t| {
                let bytes = t.as_raw_bytes()?.to_vec();
                Ok(Tensor::new(bytes, t.shape().to_vec(), t.dtype()))
            })
            .collect::<Result<Vec<_>>>()?;

        let past_values = self.past_values.iter()
            .map(|t| {
                let bytes = t.as_raw_bytes()?.to_vec();
                Ok(Tensor::new(bytes, t.shape().to_vec(), t.dtype()))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            past_keys,
            past_values,
            max_seq_len: self.max_seq_len,
            current_len: self.current_len,
            head_dim: self.head_dim,
            num_kv_heads: self.num_kv_heads,
        })
    }

    /// Create a KVCache with a parameterized batch size (for future batched inference).
    pub fn new_batched(num_layers: usize, max_seq_len: usize, head_dim: usize, num_kv_heads: usize, batch_size: usize) -> Self {
        let key_shape = vec![batch_size, num_kv_heads, max_seq_len, head_dim];
        let val_shape = vec![batch_size, num_kv_heads, max_seq_len, head_dim];

        let past_keys = (0..num_layers)
            .map(|_| Tensor::zeros(&key_shape))
            .collect();

        let past_values = (0..num_layers)
            .map(|_| Tensor::zeros(&val_shape))
            .collect();

        Self {
            past_keys,
            past_values,
            max_seq_len,
            current_len: 0,
            head_dim,
            num_kv_heads,
        }
    }
}
