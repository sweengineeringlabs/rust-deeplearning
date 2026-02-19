//! Attention mechanisms: Multi-Head Attention with GQA/RoPE/ALiBi,
//! Causal Self-Attention for GPT, and KV cache support.

use crate::api::error::{NnError, NnResult};
use crate::api::types::PositionEncoding;
use crate::core::kv_cache::KVCache;
use crate::core::linear::Linear;
use crate::core::rms_norm::RMSNorm;
use crate::core::rope::{alibi_bias, compute_alibi_slopes, RoPEFreqs};
use std::time::Instant;
use rustml_core::{DType, Tensor};

/// Multi-head attention with GQA, RoPE, ALiBi, and KV-cache support.
///
/// Supports:
/// - Standard multi-head attention (num_kv_heads == num_heads)
/// - Grouped Query Attention (num_kv_heads < num_heads)
/// - Rotary Position Encoding (RoPE)
/// - Attention with Linear Biases (ALiBi)
/// - KV caching for efficient autoregressive decoding
pub struct MultiHeadAttention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    d_model: usize,
    causal: bool,
    position_encoding: PositionEncoding,
    rope_freqs: Option<RoPEFreqs>,
    alibi_slopes: Option<Vec<f32>>,
    attn_logit_cap: Option<f32>,
    window_size: Option<usize>,
    attn_scale: Option<f32>,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,

    /// Use in-place scalar multiplication for attention score scaling (default true).
    /// When false, uses allocating `div_scalar` instead.
    use_inplace_scaling: bool,

    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,

    /// Fused Q+K+V projection for decode optimization.
    /// Created by `fuse_qkv_weights()` — reduces 3 matmul dispatches to 1.
    pub fused_qkv: Option<Linear>,
    /// Q output dimension within fused output (= num_heads * head_dim).
    fused_q_dim: usize,
    /// Single KV output dimension within fused output (= num_kv_heads * head_dim).
    fused_kv_dim: usize,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of KV heads (None = same as num_heads)
    /// * `bias` - Whether to use bias in projections
    /// * `causal` - Whether to apply causal masking
    /// * `position_encoding` - Position encoding strategy
    /// * `max_seq_len` - Maximum sequence length (for RoPE table precomputation)
    /// * `rope_theta` - RoPE theta parameter (typically 10000.0)
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        bias: bool,
        causal: bool,
        position_encoding: PositionEncoding,
        max_seq_len: usize,
        rope_theta: f32,
    ) -> NnResult<Self> {
        if d_model % num_heads != 0 {
            return Err(NnError::InvalidConfig(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            )));
        }
        let head_dim = d_model / num_heads;
        let n_kv = num_kv_heads.unwrap_or(num_heads);

        if num_heads % n_kv != 0 {
            return Err(NnError::InvalidConfig(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                num_heads, n_kv
            )));
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
            attn_logit_cap: None,
            window_size: None,
            attn_scale: None,
            q_norm: None,
            k_norm: None,
            use_inplace_scaling: true,
            q_proj: Linear::with_bias(d_model, d_model, bias),
            k_proj: Linear::with_bias(d_model, kv_dim, bias),
            v_proj: Linear::with_bias(d_model, kv_dim, bias),
            out_proj: Linear::with_bias(d_model, d_model, bias),
            fused_qkv: None,
            fused_q_dim: 0,
            fused_kv_dim: 0,
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
    ) -> NnResult<Self> {
        if d_model % num_heads != 0 {
            return Err(NnError::InvalidConfig(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            )));
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
            attn_logit_cap: None,
            window_size: None,
            attn_scale: None,
            q_norm: None,
            k_norm: None,
            use_inplace_scaling: true,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            fused_qkv: None,
            fused_q_dim: 0,
            fused_kv_dim: 0,
        })
    }

    /// Construct from pre-loaded projection layers with explicit head_dim and optional attn_scale.
    ///
    /// Used by architectures like Gemma 3 where `head_dim` is independent of `d_model / num_heads`
    /// and attention scaling uses a custom divisor instead of `sqrt(head_dim)`.
    pub fn from_weights_with_head_dim(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        head_dim: usize,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        causal: bool,
        rope_freqs: Option<RoPEFreqs>,
        attn_scale: Option<f32>,
    ) -> NnResult<Self> {
        let n_kv = num_kv_heads.unwrap_or(num_heads);

        if num_heads % n_kv != 0 {
            return Err(NnError::InvalidConfig(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                num_heads, n_kv
            )));
        }

        Ok(Self {
            num_heads,
            num_kv_heads: n_kv,
            head_dim,
            d_model,
            causal,
            position_encoding: if rope_freqs.is_some() { PositionEncoding::RoPE } else { PositionEncoding::None },
            rope_freqs,
            alibi_slopes: None,
            attn_logit_cap: None,
            window_size: None,
            attn_scale,
            q_norm: None,
            k_norm: None,
            use_inplace_scaling: true,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            fused_qkv: None,
            fused_q_dim: 0,
            fused_kv_dim: 0,
        })
    }

    /// Number of KV heads (for cache sizing).
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let (mut total, mut frozen) = (0, 0);
        for proj in [&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj] {
            let (t, f) = proj.parameter_count();
            total += t;
            frozen += f;
        }
        if let Some(ref qn) = self.q_norm {
            let (t, f) = qn.parameter_count();
            total += t;
            frozen += f;
        }
        if let Some(ref kn) = self.k_norm {
            let (t, f) = kn.parameter_count();
            total += t;
            frozen += f;
        }
        (total, frozen)
    }

    /// Toggle native Q4 integer matmul on all projection layers.
    pub fn set_native_q4_matmul(&mut self, enabled: bool) {
        self.q_proj.set_native_q4_matmul(enabled);
        self.k_proj.set_native_q4_matmul(enabled);
        self.v_proj.set_native_q4_matmul(enabled);
        self.out_proj.set_native_q4_matmul(enabled);
    }

    /// Set attention logit soft-capping (Gemma2): `cap * tanh(scores / cap)`.
    pub fn set_attn_logit_cap(&mut self, cap: f32) {
        self.attn_logit_cap = Some(cap);
    }

    /// Set sliding window size for local attention (Mistral/Gemma2).
    pub fn set_window_size(&mut self, window_size: usize) {
        self.window_size = Some(window_size);
    }

    /// Set QK normalization (Gemma 3): RMSNorm applied to Q and K after projection.
    pub fn set_qk_norms(&mut self, q_norm: RMSNorm, k_norm: RMSNorm) {
        self.q_norm = Some(q_norm);
        self.k_norm = Some(k_norm);
    }

    /// Toggle in-place attention score scaling.
    /// When true (default), uses `mul_scalar_inplace` (zero-alloc).
    /// When false, uses allocating `div_scalar`.
    pub fn set_use_inplace_scaling(&mut self, enabled: bool) {
        self.use_inplace_scaling = enabled;
    }

    /// Fuse q_proj, k_proj, v_proj Q8_0 weights into a single `[q_dim+kv_dim+kv_dim, in_features]` tensor.
    /// Reduces 3 matmul dispatches to 1 during decode. No-op if weights are not all Q8_0 or biases are present.
    pub fn fuse_qkv_weights(&mut self) -> bool {
        if self.q_proj.weight.dtype() != DType::Q8_0 { return false; }
        if self.k_proj.weight.dtype() != DType::Q8_0 { return false; }
        if self.v_proj.weight.dtype() != DType::Q8_0 { return false; }
        if self.q_proj.bias.is_some() || self.k_proj.bias.is_some() || self.v_proj.bias.is_some() { return false; }
        if self.q_proj.in_features != self.k_proj.in_features { return false; }
        if self.q_proj.in_features != self.v_proj.in_features { return false; }

        let q_bytes = match self.q_proj.weight.as_raw_bytes() { Ok(b) => b, Err(_) => return false };
        let k_bytes = match self.k_proj.weight.as_raw_bytes() { Ok(b) => b, Err(_) => return false };
        let v_bytes = match self.v_proj.weight.as_raw_bytes() { Ok(b) => b, Err(_) => return false };

        let mut fused_bytes = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
        fused_bytes.extend_from_slice(q_bytes);
        fused_bytes.extend_from_slice(k_bytes);
        fused_bytes.extend_from_slice(v_bytes);

        let in_features = self.q_proj.in_features;
        let out_features = self.q_proj.out_features + self.k_proj.out_features + self.v_proj.out_features;

        let fused_weight = Tensor::new(fused_bytes, vec![out_features, in_features], DType::Q8_0);
        self.fused_qkv = Some(Linear {
            weight: fused_weight,
            bias: None,
            in_features,
            out_features,
            frozen: true,
            use_native_q4: false,
        });
        self.fused_q_dim = self.q_proj.out_features;
        self.fused_kv_dim = self.k_proj.out_features;
        true
    }

    /// Forward pass without KV cache.
    ///
    /// Input: [B, S, d_model] -> Output: [B, S, d_model]
    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Project inputs
        let (q, k, v) = if let Some(ref fused) = self.fused_qkv {
            let fused_out = fused.forward(input)?;
            let q_end = self.fused_q_dim;
            let k_end = q_end + self.fused_kv_dim;
            let v_end = k_end + self.fused_kv_dim;
            let q = fused_out.slice(-1, 0, q_end)?.contiguous()?;
            let k = fused_out.slice(-1, q_end, k_end)?.contiguous()?;
            let v = fused_out.slice(-1, k_end, v_end)?.contiguous()?;
            (q, k, v)
        } else {
            let q = self.q_proj.forward(input)?;
            let k = self.k_proj.forward(input)?;
            let v = self.v_proj.forward(input)?;
            (q, k, v)
        };

        // Reshape Q: [B, S, num_heads, D] -> [B, num_heads, S, D]
        let q = q
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Reshape K/V: [B, S, num_kv_heads, D] -> [B, num_kv_heads, S, D]
        let k = k
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        // QK normalization (Gemma 3)
        let q = if let Some(ref qn) = self.q_norm { qn.forward(&q)? } else { q };
        let k = if let Some(ref kn) = self.k_norm { kn.forward(&k)? } else { k };

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

        // Attention scores: Q @ K^T / scale
        let k_t = k.transpose(2, 3)?;
        let mut scores = q.matmul(&k_t)?;
        let scale = self.attn_scale.unwrap_or_else(|| (self.head_dim as f32).sqrt());
        if self.use_inplace_scaling {
            // In-place scaling avoids allocating a new tensor (matmul result has refcount 1)
            scores.mul_scalar_inplace(1.0 / scale)?;
        } else {
            scores = scores.div_scalar(scale);
        }

        // Logit soft-capping
        let scores = if let Some(cap) = self.attn_logit_cap {
            scores.div_scalar(cap).tanh().mul_scalar(cap)
        } else {
            scores
        };

        // Apply mask/bias
        let scores = if let Some(ref slopes) = self.alibi_slopes {
            let bias = alibi_bias(slopes, seq_len, seq_len, self.causal);
            scores.add(&bias)?
        } else if self.causal && seq_len > 1 {
            if let Some(ws) = self.window_size {
                let mask = Tensor::sliding_window_mask(seq_len, seq_len, ws);
                scores.add(&mask)?
            } else {
                let mask = Tensor::causal_mask(seq_len, seq_len);
                scores.add(&mask)?
            }
        } else {
            scores
        };

        let attn = scores.softmax(-1)?;
        let context = attn.matmul(&v)?;

        let context = context
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        self.out_proj.forward(&context)
    }

    /// Forward pass with KV cache for autoregressive decoding.
    pub fn forward_with_cache(
        &self,
        input: &Tensor,
        cache: &mut KVCache,
        layer_idx: usize,
    ) -> NnResult<Tensor> {
        let _t = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
        let result = self.forward_with_cache_inner(input, cache, layer_idx);
        if let Some(t) = _t {
            log::debug!("[perf] attention::forward layer={} {:?} {:.3}ms",
                layer_idx, input.shape(), t.elapsed().as_secs_f64() * 1000.0);
        }
        result
    }

    fn forward_with_cache_inner(
        &self,
        input: &Tensor,
        cache: &mut KVCache,
        layer_idx: usize,
    ) -> NnResult<Tensor> {
        let trace = log::log_enabled!(log::Level::Trace);

        if cache.head_dim() != self.head_dim {
            return Err(NnError::InvalidConfig(format!(
                "KVCache head_dim ({}) does not match attention head_dim ({})",
                cache.head_dim(),
                self.head_dim
            )));
        }

        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let start_pos = cache.current_len;

        // Project
        let t_proj = if trace { Some(Instant::now()) } else { None };
        let (q, k, v) = if let Some(ref fused) = self.fused_qkv {
            let fused_out = fused.forward(input)?;
            let q_end = self.fused_q_dim;
            let k_end = q_end + self.fused_kv_dim;
            let v_end = k_end + self.fused_kv_dim;
            let q = fused_out.slice(-1, 0, q_end)?.contiguous()?;
            let k = fused_out.slice(-1, q_end, k_end)?.contiguous()?;
            let v = fused_out.slice(-1, k_end, v_end)?.contiguous()?;
            (q, k, v)
        } else {
            let q = self.q_proj.forward(input)?;
            let k = self.k_proj.forward(input)?;
            let v = self.v_proj.forward(input)?;
            (q, k, v)
        };
        let proj_ms = t_proj.map(|t| t.elapsed().as_secs_f64() * 1000.0);

        // Reshape Q: [B, S, num_heads, D] -> [B, num_heads, S, D]
        let q = q
            .reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Reshape K/V: [B, S, num_kv_heads, D] -> [B, num_kv_heads, S, D]
        let k = k
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        // QK normalization (Gemma 3)
        let t_norm = if trace { Some(Instant::now()) } else { None };
        let q = if let Some(ref qn) = self.q_norm { qn.forward(&q)? } else { q };
        let k = if let Some(ref kn) = self.k_norm { kn.forward(&k)? } else { k };
        let norm_ms = t_norm.map(|t| t.elapsed().as_secs_f64() * 1000.0);

        // Apply RoPE before cache update
        let t_rope = if trace { Some(Instant::now()) } else { None };
        let (q, k) = if let Some(ref rope) = self.rope_freqs {
            (rope.apply(&q, start_pos)?, rope.apply(&k, start_pos)?)
        } else {
            (q, k)
        };
        let rope_ms = t_rope.map(|t| t.elapsed().as_secs_f64() * 1000.0);

        // Update cache
        cache.update(layer_idx, k.clone(), v.clone())?;

        // Get full K/V history
        let total_len = cache.current_len + seq_len;
        let (k_full, v_full) = cache.get_view(layer_idx, total_len)?;

        // GQA: expand K/V
        let n_rep = self.num_heads / self.num_kv_heads;
        let k_full = k_full.repeat_kv(n_rep)?;
        let v_full = v_full.repeat_kv(n_rep)?;

        // Attention scores: Q @ K^T
        let t_qkt = if trace { Some(Instant::now()) } else { None };
        let k_t = k_full.transpose(2, 3)?;
        let mut scores = q.matmul(&k_t)?;
        let scale = self.attn_scale.unwrap_or_else(|| (self.head_dim as f32).sqrt());
        if self.use_inplace_scaling {
            // In-place scaling avoids allocating a new tensor (matmul result has refcount 1)
            scores.mul_scalar_inplace(1.0 / scale)?;
        } else {
            scores = scores.div_scalar(scale);
        }
        let qkt_ms = t_qkt.map(|t| t.elapsed().as_secs_f64() * 1000.0);

        // Logit soft-capping
        let scores = if let Some(cap) = self.attn_logit_cap {
            scores.div_scalar(cap).tanh().mul_scalar(cap)
        } else {
            scores
        };

        // Apply mask/bias
        let scores = if let Some(ref slopes) = self.alibi_slopes {
            let bias = alibi_bias(slopes, seq_len, total_len, self.causal);
            scores.add(&bias)?
        } else if self.causal && seq_len > 1 {
            if let Some(ws) = self.window_size {
                let mask = Tensor::sliding_window_mask(seq_len, total_len, ws);
                scores.add(&mask)?
            } else {
                let mask = Tensor::causal_mask(seq_len, total_len);
                scores.add(&mask)?
            }
        } else {
            scores
        };

        // Softmax
        let t_softmax = if trace { Some(Instant::now()) } else { None };
        let attn = scores.softmax(-1)?;
        let softmax_ms = t_softmax.map(|t| t.elapsed().as_secs_f64() * 1000.0);

        // Attention @ V
        let t_av = if trace { Some(Instant::now()) } else { None };
        let context = attn.matmul(&v_full)?;
        let av_ms = t_av.map(|t| t.elapsed().as_secs_f64() * 1000.0);

        let context = context
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        // Output projection
        let t_out = if trace { Some(Instant::now()) } else { None };
        let output = self.out_proj.forward(&context)?;
        let out_ms = t_out.map(|t| t.elapsed().as_secs_f64() * 1000.0);

        if trace {
            log::trace!(
                "[attn] layer={} QKV={:.3}ms QKnorm={:.3}ms RoPE={:.3}ms QK^T={:.3}ms softmax={:.3}ms A*V={:.3}ms out={:.3}ms",
                layer_idx,
                proj_ms.unwrap_or(0.0),
                norm_ms.unwrap_or(0.0),
                rope_ms.unwrap_or(0.0),
                qkt_ms.unwrap_or(0.0),
                softmax_ms.unwrap_or(0.0),
                av_ms.unwrap_or(0.0),
                out_ms.unwrap_or(0.0),
            );
        }

        Ok(output)
    }
}

/// Causal Self-Attention for GPT-style models
///
/// This implements the causal (autoregressive) attention mechanism used in GPT-2,
/// where each position can only attend to previous positions.
///
/// Key features:
/// - Combined QKV projection (GPT-2 style: c_attn)
/// - Causal masking to prevent attending to future tokens
/// - Multi-head attention with proper reshaping
#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    /// Combined QKV projection [3 * n_embd, n_embd]
    pub c_attn: Linear,
    /// Output projection
    pub c_proj: Linear,
    /// Number of attention heads
    pub n_head: usize,
    /// Embedding dimension
    pub n_embd: usize,
}

impl CausalSelfAttention {
    /// Create a new causal self-attention layer
    pub fn new(n_embd: usize, n_head: usize) -> Self {
        assert!(
            n_embd % n_head == 0,
            "Embedding dimension must be divisible by number of heads"
        );

        // Combined QKV projection (GPT-2 style)
        let c_attn = Linear::new(n_embd, 3 * n_embd);
        let c_proj = Linear::new(n_embd, n_embd);

        Self {
            c_attn,
            c_proj,
            n_head,
            n_embd,
        }
    }

    /// Create from pre-trained weights
    pub fn from_weights(
        c_attn_weight: Tensor,
        c_attn_bias: Option<Tensor>,
        c_proj_weight: Tensor,
        c_proj_bias: Option<Tensor>,
        n_head: usize,
    ) -> NnResult<Self> {
        let n_embd = c_proj_weight.shape()[0];

        let c_attn = Linear::from_weights(c_attn_weight, c_attn_bias)?;
        let c_proj = Linear::from_weights(c_proj_weight, c_proj_bias)?;

        Ok(Self {
            c_attn,
            c_proj,
            n_head,
            n_embd,
        })
    }

    /// Forward pass
    ///
    /// Input shape: [batch_size, seq_len, n_embd]
    /// Output shape: [batch_size, seq_len, n_embd]
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(NnError::ShapeMismatch(format!(
                "Expected 3D input [B, T, C], got {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let n_embd = shape[2];
        let head_dim = n_embd / self.n_head;

        // 1. Combined QKV projection
        let qkv = self.c_attn.forward(x)?; // [B, T, 3*C]

        // 2. Split into Q, K, V
        let q = qkv.slice(-1, 0, n_embd)?;
        let k = qkv.slice(-1, n_embd, 2 * n_embd)?;
        let v = qkv.slice(-1, 2 * n_embd, 3 * n_embd)?;

        // 3. Reshape to multi-head: [B, T, C] -> [B, H, T, C/H]
        let q = q
            .reshape(&[batch_size, seq_len, self.n_head, head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape(&[batch_size, seq_len, self.n_head, head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape(&[batch_size, seq_len, self.n_head, head_dim])?
            .transpose(1, 2)?;

        // 4. Compute attention scores: Q @ K^T / sqrt(d_k)
        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(&k.t()?)?.div_scalar(scale); // [B, H, T, T]

        // 5. Create and apply causal mask
        // Mask is 0 for positions to keep, 1 for positions to mask
        let causal_mask = Self::create_causal_mask(seq_len);
        let scores = scores.masked_fill(&causal_mask, f32::NEG_INFINITY)?;

        // 6. Softmax to get attention weights
        let attn = scores.softmax(-1)?;

        // 7. Compute weighted values: attn @ V
        let out = attn.matmul(&v)?; // [B, H, T, C/H]

        // 8. Reshape back: [B, H, T, C/H] -> [B, T, C]
        let out = out
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, n_embd])?;

        // 9. Output projection
        self.c_proj.forward(&out)
    }

    /// Create a causal mask for the given sequence length
    ///
    /// Returns a mask where future positions are 1.0 (to be masked)
    /// and current/past positions are 0.0 (to be kept)
    ///
    /// Example for seq_len=4:
    /// ```text
    /// [0, 1, 1, 1]  <- position 0 can only see itself
    /// [0, 0, 1, 1]  <- position 1 can see 0 and itself
    /// [0, 0, 0, 1]  <- position 2 can see 0, 1, and itself
    /// [0, 0, 0, 0]  <- position 3 can see all
    /// ```
    fn create_causal_mask(seq_len: usize) -> Tensor {
        let tril = Tensor::tril(seq_len);
        // Invert: where tril is 1, mask is 0; where tril is 0, mask is 1
        let ones = Tensor::ones(vec![seq_len, seq_len]);
        ones.sub(&tril).unwrap()
    }
}

impl crate::api::traits::Attention for CausalSelfAttention {
    fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        CausalSelfAttention::forward(self, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_attention_shape() {
        let attn = CausalSelfAttention::new(768, 12);
        let x = Tensor::randn(vec![2, 10, 768]);
        let y = attn.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10, 768]);
    }

    #[test]
    fn test_causal_mask() {
        let mask = CausalSelfAttention::create_causal_mask(4);
        // Position (0, 0) should be 0 (can attend to self)
        assert_eq!(mask.get(&[0, 0]).unwrap(), 0.0);
        // Position (0, 1) should be 1 (cannot attend to future)
        assert_eq!(mask.get(&[0, 1]).unwrap(), 1.0);
        // Position (3, 0) should be 0 (can attend to past)
        assert_eq!(mask.get(&[3, 0]).unwrap(), 0.0);
        // Position (3, 3) should be 0 (can attend to self)
        assert_eq!(mask.get(&[3, 3]).unwrap(), 0.0);
    }

    #[test]
    fn test_multi_head_attention_basic() {
        let mha = MultiHeadAttention::new(
            64, 4, None, true, false,
            PositionEncoding::None, 128, 10000.0,
        )
        .unwrap();
        let x = Tensor::randn(vec![2, 8, 64]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_multi_head_attention_causal() {
        let mha = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::None, 128, 10000.0,
        )
        .unwrap();
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_multi_head_attention_rope() {
        let mha = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::RoPE, 128, 10000.0,
        )
        .unwrap();
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_multi_head_attention_gqa() {
        // 8 Q heads, 2 KV heads -> GQA ratio = 4
        let mha = MultiHeadAttention::new(
            64, 8, Some(2), false, true,
            PositionEncoding::RoPE, 128, 10000.0,
        )
        .unwrap();
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_multi_head_attention_alibi() {
        let mha = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::ALiBi, 128, 10000.0,
        )
        .unwrap();
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_multi_head_attention_logit_cap() {
        let mut mha = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::None, 128, 10000.0,
        )
        .unwrap();
        mha.set_attn_logit_cap(50.0);
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_multi_head_attention_sliding_window() {
        let mut mha = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::RoPE, 128, 10000.0,
        )
        .unwrap();
        mha.set_window_size(4);
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_multi_head_attention_explicit_head_dim() {
        // Gemma 3 style: d_model=128, num_heads=4, head_dim=64 (decoupled)
        // q_proj: [128, 4*64=256], k_proj: [128, 2*64=128], etc.
        let d_model = 128;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_proj = Linear::new_no_bias(d_model, q_dim);
        let k_proj = Linear::new_no_bias(d_model, kv_dim);
        let v_proj = Linear::new_no_bias(d_model, kv_dim);
        let out_proj = Linear::new_no_bias(q_dim, d_model);

        let rope = RoPEFreqs::new(head_dim, 128, 10000.0);
        let mha = MultiHeadAttention::from_weights_with_head_dim(
            d_model, num_heads, Some(num_kv_heads), head_dim,
            q_proj, k_proj, v_proj, out_proj,
            true, Some(rope), None,
        ).unwrap();

        let x = Tensor::randn(vec![1, 8, d_model]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, d_model]);
    }

    #[test]
    fn test_multi_head_attention_custom_attn_scale() {
        // Test custom attn_scale (Gemma 3: query_pre_attn_scalar=256 → sqrt(256)=16)
        let d_model = 64;
        let num_heads = 4;
        let head_dim = d_model / num_heads;

        let q_proj = Linear::new_no_bias(d_model, d_model);
        let k_proj = Linear::new_no_bias(d_model, d_model);
        let v_proj = Linear::new_no_bias(d_model, d_model);
        let out_proj = Linear::new_no_bias(d_model, d_model);

        let mha = MultiHeadAttention::from_weights_with_head_dim(
            d_model, num_heads, None, head_dim,
            q_proj, k_proj, v_proj, out_proj,
            true, None, Some(16.0), // custom scale
        ).unwrap();

        let x = Tensor::randn(vec![1, 4, d_model]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 4, d_model]);
    }

    #[test]
    fn test_multi_head_attention_sliding_window_with_cache() {
        use crate::core::kv_cache::KVCache;
        let mut mha = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::RoPE, 128, 10000.0,
        )
        .unwrap();
        mha.set_window_size(4);
        let mut cache = KVCache::new(1, 128, 16, 4);
        // Prefill with 6 tokens
        let x = Tensor::randn(vec![1, 6, 64]);
        let y = mha.forward_with_cache(&x, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 6, 64]);
        // Decode step: 1 token
        let x2 = Tensor::randn(vec![1, 1, 64]);
        let y2 = mha.forward_with_cache(&x2, &mut cache, 0).unwrap();
        assert_eq!(y2.shape(), &[1, 1, 64]);
    }

    #[test]
    fn test_multi_head_attention_qk_norms() {
        use crate::core::rms_norm::RMSNorm;
        let d_model = 64;
        let num_heads = 4;
        let head_dim = 16;

        let mut mha = MultiHeadAttention::new(
            d_model, num_heads, None, false, true,
            PositionEncoding::RoPE, 128, 10000.0,
        ).unwrap();

        mha.set_qk_norms(
            RMSNorm::new(head_dim, 1e-6),
            RMSNorm::new(head_dim, 1e-6),
        );

        let x = Tensor::randn(vec![1, 8, d_model]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, d_model]);

        // Also test with cache
        let mut cache = KVCache::new(1, 128, head_dim, num_heads);
        let y2 = mha.forward_with_cache(&x, &mut cache, 0).unwrap();
        assert_eq!(y2.shape(), &[1, 8, d_model]);
    }

    // ==================== In-place score scaling tests ====================

    #[test]
    fn test_attention_inplace_scale_forward() {
        // Verifies that mul_scalar_inplace(1/scale) produces finite, valid output
        let mha = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::None, 128, 10000.0,
        ).unwrap();
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
        let flat = y.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "forward output contains NaN/Inf");
    }

    #[test]
    fn test_attention_inplace_scale_with_cache_decode() {
        // Exercises the in-place scaling path during decode (seq_len=1)
        let mha = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::RoPE, 128, 10000.0,
        ).unwrap();

        let mut cache = KVCache::new(1, 128, 16, 4);

        // Prefill
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = mha.forward_with_cache(&x, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
        cache.advance(8);

        // Decode — in-place mul_scalar_inplace(1/scale) on fresh matmul result
        let x2 = Tensor::randn(vec![1, 1, 64]);
        let y2 = mha.forward_with_cache(&x2, &mut cache, 0).unwrap();
        assert_eq!(y2.shape(), &[1, 1, 64]);
        let flat = y2.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "decode output contains NaN/Inf");
    }

    #[test]
    fn test_attention_inplace_scale_gqa_with_cache() {
        // GQA with in-place scaling (8 Q heads, 2 KV heads)
        let mha = MultiHeadAttention::new(
            64, 8, Some(2), false, true,
            PositionEncoding::RoPE, 128, 10000.0,
        ).unwrap();

        let mut cache = KVCache::new(1, 128, 8, 2);

        let x = Tensor::randn(vec![1, 6, 64]);
        let y = mha.forward_with_cache(&x, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 6, 64]);
        cache.advance(6);

        let x2 = Tensor::randn(vec![1, 1, 64]);
        let y2 = mha.forward_with_cache(&x2, &mut cache, 0).unwrap();
        assert_eq!(y2.shape(), &[1, 1, 64]);
        let flat = y2.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_attention_inplace_scale_with_logit_cap() {
        // Logit capping still uses div_scalar (allocating) — verify it still works
        let mut mha = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::None, 128, 10000.0,
        ).unwrap();
        mha.set_attn_logit_cap(50.0);

        let mut cache = KVCache::new(1, 128, 16, 4);
        let x = Tensor::randn(vec![1, 4, 64]);
        let y = mha.forward_with_cache(&x, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 4, 64]);
        let flat = y.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_attention_inplace_scale_custom_attn_scale() {
        // Custom attn_scale (Gemma 3 style) — still uses inplace scaling
        let d_model = 64;
        let num_heads = 4;
        let head_dim = d_model / num_heads;

        let q_proj = Linear::new_no_bias(d_model, d_model);
        let k_proj = Linear::new_no_bias(d_model, d_model);
        let v_proj = Linear::new_no_bias(d_model, d_model);
        let out_proj = Linear::new_no_bias(d_model, d_model);

        let mha = MultiHeadAttention::from_weights_with_head_dim(
            d_model, num_heads, None, head_dim,
            q_proj, k_proj, v_proj, out_proj,
            true, None, Some(16.0),
        ).unwrap();

        let mut cache = KVCache::new(1, 64, head_dim, num_heads);
        let x = Tensor::randn(vec![1, 4, d_model]);
        let y = mha.forward_with_cache(&x, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 4, d_model]);
        let flat = y.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    // ==================== Configurable inplace scaling tests ====================

    #[test]
    fn test_attention_inplace_vs_allocating_scaling_match() {
        // Same weights, same input → inplace and allocating scaling must match
        let d_model = 64;
        let num_heads = 4;

        let q_proj = Linear::new_no_bias(d_model, d_model);
        let k_proj = Linear::new_no_bias(d_model, d_model);
        let v_proj = Linear::new_no_bias(d_model, d_model);
        let out_proj = Linear::new_no_bias(d_model, d_model);

        let mut mha_inplace = MultiHeadAttention::from_weights(
            d_model, num_heads, None,
            q_proj.clone(), k_proj.clone(), v_proj.clone(), out_proj.clone(),
            true, PositionEncoding::None, 128, 10000.0,
        ).unwrap();
        mha_inplace.set_use_inplace_scaling(true);

        let mut mha_alloc = MultiHeadAttention::from_weights(
            d_model, num_heads, None,
            q_proj, k_proj, v_proj, out_proj,
            true, PositionEncoding::None, 128, 10000.0,
        ).unwrap();
        mha_alloc.set_use_inplace_scaling(false);

        let x = Tensor::randn(vec![1, 6, d_model]);
        let y_inplace = mha_inplace.forward(&x).unwrap();
        let y_alloc = mha_alloc.forward(&x).unwrap();

        let d1 = y_inplace.as_slice_f32().unwrap();
        let d2 = y_alloc.as_slice_f32().unwrap();
        assert_eq!(d1.len(), d2.len());
        for i in 0..d1.len() {
            assert!((d1[i] - d2[i]).abs() < 1e-5,
                "inplace vs alloc scaling mismatch at {}: {} vs {}", i, d1[i], d2[i]);
        }
    }

    #[test]
    fn test_attention_inplace_vs_allocating_with_cache() {
        let d_model = 64;
        let num_heads = 4;
        let head_dim = d_model / num_heads;

        let q_proj = Linear::new_no_bias(d_model, d_model);
        let k_proj = Linear::new_no_bias(d_model, d_model);
        let v_proj = Linear::new_no_bias(d_model, d_model);
        let out_proj = Linear::new_no_bias(d_model, d_model);

        let mut mha_inplace = MultiHeadAttention::from_weights(
            d_model, num_heads, None,
            q_proj.clone(), k_proj.clone(), v_proj.clone(), out_proj.clone(),
            true, PositionEncoding::None, 128, 10000.0,
        ).unwrap();
        mha_inplace.set_use_inplace_scaling(true);

        let mut mha_alloc = MultiHeadAttention::from_weights(
            d_model, num_heads, None,
            q_proj, k_proj, v_proj, out_proj,
            true, PositionEncoding::None, 128, 10000.0,
        ).unwrap();
        mha_alloc.set_use_inplace_scaling(false);

        let x = Tensor::randn(vec![1, 4, d_model]);

        let mut cache1 = KVCache::new(1, 128, head_dim, num_heads);
        let mut cache2 = KVCache::new(1, 128, head_dim, num_heads);

        let y1 = mha_inplace.forward_with_cache(&x, &mut cache1, 0).unwrap();
        let y2 = mha_alloc.forward_with_cache(&x, &mut cache2, 0).unwrap();

        let d1 = y1.as_slice_f32().unwrap();
        let d2 = y2.as_slice_f32().unwrap();
        for i in 0..d1.len() {
            assert!((d1[i] - d2[i]).abs() < 1e-5,
                "cache inplace vs alloc mismatch at {}: {} vs {}", i, d1[i], d2[i]);
        }
    }
}
