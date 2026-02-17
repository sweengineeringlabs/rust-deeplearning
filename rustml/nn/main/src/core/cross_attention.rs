//! Cross-attention: Q from decoder, K/V from encoder.
//! No causal mask. Supports GQA via repeat_kv.

use crate::api::error::{NnError, NnResult};
use crate::core::kv_cache::KVCache;
use crate::core::linear::Linear;
use rustml_core::Tensor;

/// Cross-attention layer for encoder-decoder architectures.
pub struct CrossAttention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    d_model: usize,
    encoder_dim: usize,

    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
}

impl CrossAttention {
    pub fn new(
        d_model: usize,
        encoder_dim: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        bias: bool,
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

        Ok(Self {
            num_heads,
            num_kv_heads: n_kv,
            head_dim,
            d_model,
            encoder_dim,
            q_proj: Linear::with_bias(d_model, d_model, bias),
            k_proj: Linear::with_bias(encoder_dim, kv_dim, bias),
            v_proj: Linear::with_bias(encoder_dim, kv_dim, bias),
            out_proj: Linear::with_bias(d_model, d_model, bias),
        })
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

    /// Forward pass: decoder_input [B, S_dec, d_model], encoder_output [B, S_enc, encoder_dim]
    pub fn forward(
        &self,
        decoder_input: &Tensor,
        encoder_output: &Tensor,
    ) -> NnResult<Tensor> {
        let batch_size = decoder_input.shape()[0];
        let dec_len = decoder_input.shape()[1];
        let enc_len = encoder_output.shape()[1];

        let q = self.q_proj.forward(decoder_input)?;
        let k = self.k_proj.forward(encoder_output)?;
        let v = self.v_proj.forward(encoder_output)?;

        let q = q
            .reshape(&[batch_size, dec_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape(&[batch_size, enc_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape(&[batch_size, enc_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        // GQA: expand K/V to match Q heads
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = k.repeat_kv(n_rep)?;
        let v = v.repeat_kv(n_rep)?;

        let k_t = k.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?;
        let scale = (self.head_dim as f32).sqrt();
        let scores = scores.div_scalar(scale);

        // No causal mask for cross-attention
        let attn = scores.softmax(-1)?;
        let context = attn.matmul(&v)?;

        let context = context
            .transpose(1, 2)?
            .reshape(&[batch_size, dec_len, self.d_model])?;

        self.out_proj.forward(&context)
    }

    /// Forward with KV cache for encoder outputs.
    ///
    /// On first call (`encoder_cached=false`), K/V are projected from `encoder_output`
    /// and stored in the cache. On subsequent calls (`encoder_cached=true`), K/V are
    /// read from the cache.
    pub fn forward_with_cache(
        &self,
        decoder_input: &Tensor,
        encoder_output: &Tensor,
        cache: &mut KVCache,
        layer_idx: usize,
        encoder_cached: bool,
    ) -> NnResult<Tensor> {
        let batch_size = decoder_input.shape()[0];
        let dec_len = decoder_input.shape()[1];

        let q = self.q_proj.forward(decoder_input)?;
        let q = q
            .reshape(&[batch_size, dec_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let (k_full, v_full) = if encoder_cached {
            let total = cache.current_len;
            cache.get_view(layer_idx, total)?
        } else {
            let enc_len = encoder_output.shape()[1];
            let k = self.k_proj.forward(encoder_output)?;
            let v = self.v_proj.forward(encoder_output)?;

            let k = k
                .reshape(&[batch_size, enc_len, self.num_kv_heads, self.head_dim])?
                .transpose(1, 2)?;
            let v = v
                .reshape(&[batch_size, enc_len, self.num_kv_heads, self.head_dim])?
                .transpose(1, 2)?;

            cache.update(layer_idx, k.clone(), v.clone())?;
            let total = cache.current_len + enc_len;
            cache.get_view(layer_idx, total)?
        };

        // GQA
        let n_rep = self.num_heads / self.num_kv_heads;
        let k_full = k_full.repeat_kv(n_rep)?;
        let v_full = v_full.repeat_kv(n_rep)?;

        let k_t = k_full.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?;
        let scale = (self.head_dim as f32).sqrt();
        let scores = scores.div_scalar(scale);

        let attn = scores.softmax(-1)?;
        let context = attn.matmul(&v_full)?;

        let context = context
            .transpose(1, 2)?
            .reshape(&[batch_size, dec_len, self.d_model])?;

        self.out_proj.forward(&context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_attention_shape() {
        let ca = CrossAttention::new(64, 64, 4, None, true).unwrap();
        let decoder = Tensor::randn(vec![1, 5, 64]);
        let encoder = Tensor::randn(vec![1, 10, 64]);
        let out = ca.forward(&decoder, &encoder).unwrap();
        assert_eq!(out.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cross_attention_gqa() {
        // 8 Q heads, 2 KV heads -> GQA ratio = 4
        let ca = CrossAttention::new(64, 64, 8, Some(2), false).unwrap();
        let decoder = Tensor::randn(vec![1, 4, 64]);
        let encoder = Tensor::randn(vec![1, 8, 64]);
        let out = ca.forward(&decoder, &encoder).unwrap();
        assert_eq!(out.shape(), &[1, 4, 64]);
    }
}
