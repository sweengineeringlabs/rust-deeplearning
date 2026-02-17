//! Transformer block: pre-norm architecture with self-attention + optional cross-attention + FFN.

use crate::api::error::NnResult;
use crate::api::types::PositionEncoding;
use crate::core::attention::MultiHeadAttention;
use crate::core::cross_attention::CrossAttention;
use crate::core::feed_forward::FeedForward;
use crate::core::kv_cache::KVCache;
use crate::core::layer_norm::LayerNorm;
use crate::core::rms_norm::RMSNorm;
use rustml_core::Tensor;

/// Normalization layer: either standard LayerNorm or RMSNorm.
pub enum NormLayer {
    LayerNorm(LayerNorm),
    RMSNorm(RMSNorm),
}

impl NormLayer {
    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        match self {
            NormLayer::LayerNorm(ln) => ln.forward(input),
            NormLayer::RMSNorm(rn) => rn.forward(input),
        }
    }

    pub fn parameter_count(&self) -> (usize, usize) {
        match self {
            NormLayer::LayerNorm(ln) => {
                let total = ln.weight.numel() + ln.bias.numel();
                (total, 0)
            }
            NormLayer::RMSNorm(rn) => rn.parameter_count(),
        }
    }
}

/// A single transformer block with pre-norm architecture.
///
/// Structure: x -> norm1 -> self_attn -> + -> [norm_cross -> cross_attn -> +] -> norm2 -> ffn -> +
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub cross_attention: Option<CrossAttention>,
    pub cross_attention_norm: Option<NormLayer>,
    pub feed_forward: FeedForward,
    pub attention_norm: NormLayer,
    pub ffn_norm: NormLayer,
}

impl TransformerBlock {
    /// Create a new transformer block with LayerNorm.
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: Option<usize>,
        hidden_dim: usize,
        bias: bool,
        eps: f32,
        causal: bool,
        position_encoding: PositionEncoding,
        max_seq_len: usize,
        rope_theta: f32,
    ) -> NnResult<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(
                d_model,
                num_heads,
                num_kv_heads,
                bias,
                causal,
                position_encoding,
                max_seq_len,
                rope_theta,
            )?,
            cross_attention: None,
            cross_attention_norm: None,
            feed_forward: FeedForward::new(d_model, hidden_dim, bias),
            attention_norm: NormLayer::LayerNorm(LayerNorm::with_eps(d_model, eps)),
            ffn_norm: NormLayer::LayerNorm(LayerNorm::with_eps(d_model, eps)),
        })
    }

    /// Construct from pre-loaded components (LayerNorm variant).
    pub fn from_weights(
        attention: MultiHeadAttention,
        feed_forward: FeedForward,
        attention_norm: LayerNorm,
        ffn_norm: LayerNorm,
    ) -> Self {
        Self {
            attention,
            cross_attention: None,
            cross_attention_norm: None,
            feed_forward,
            attention_norm: NormLayer::LayerNorm(attention_norm),
            ffn_norm: NormLayer::LayerNorm(ffn_norm),
        }
    }

    /// Construct from pre-loaded components (RMSNorm variant, for Llama).
    pub fn from_weights_rms(
        attention: MultiHeadAttention,
        feed_forward: FeedForward,
        attention_norm: RMSNorm,
        ffn_norm: RMSNorm,
    ) -> Self {
        Self {
            attention,
            cross_attention: None,
            cross_attention_norm: None,
            feed_forward,
            attention_norm: NormLayer::RMSNorm(attention_norm),
            ffn_norm: NormLayer::RMSNorm(ffn_norm),
        }
    }

    /// Construct with cross-attention support.
    pub fn from_weights_with_cross(
        attention: MultiHeadAttention,
        cross_attention: CrossAttention,
        cross_attention_norm: LayerNorm,
        feed_forward: FeedForward,
        attention_norm: LayerNorm,
        ffn_norm: LayerNorm,
    ) -> Self {
        Self {
            attention,
            cross_attention: Some(cross_attention),
            cross_attention_norm: Some(NormLayer::LayerNorm(cross_attention_norm)),
            feed_forward,
            attention_norm: NormLayer::LayerNorm(attention_norm),
            ffn_norm: NormLayer::LayerNorm(ffn_norm),
        }
    }

    /// Access the self-attention layer (e.g. for cache sizing queries).
    pub fn attention(&self) -> &MultiHeadAttention {
        &self.attention
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let (mut total, mut frozen) = (0, 0);

        let (t, f) = self.attention.parameter_count();
        total += t;
        frozen += f;

        let (t, f) = self.feed_forward.parameter_count();
        total += t;
        frozen += f;

        let (t, f) = self.attention_norm.parameter_count();
        total += t;
        frozen += f;

        let (t, f) = self.ffn_norm.parameter_count();
        total += t;
        frozen += f;

        if let Some(ref cross_attn) = self.cross_attention {
            let (t, f) = cross_attn.parameter_count();
            total += t;
            frozen += f;
        }
        if let Some(ref cross_norm) = self.cross_attention_norm {
            let (t, f) = cross_norm.parameter_count();
            total += t;
            frozen += f;
        }

        (total, frozen)
    }

    /// Toggle native Q4 integer matmul on all Linear layers in this block.
    pub fn set_native_q4_matmul(&mut self, enabled: bool) {
        self.attention.set_native_q4_matmul(enabled);
        self.feed_forward.set_native_q4_matmul(enabled);
    }

    /// Forward pass without KV cache.
    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        // Pre-norm architecture: x = x + attn(ln(x))
        let norm_1 = self.attention_norm.forward(input)?;
        let attn_out = self.attention.forward(&norm_1)?;
        let x = input.add(&attn_out)?;

        // x = x + ffn(ln(x))
        let norm_2 = self.ffn_norm.forward(&x)?;
        let ffn_out = self.feed_forward.forward(&norm_2)?;
        x.add(&ffn_out).map_err(Into::into)
    }

    /// Forward pass with KV cache for autoregressive decoding.
    pub fn forward_with_cache(
        &self,
        input: &Tensor,
        encoder_output: Option<&Tensor>,
        cache: &mut KVCache,
        layer_idx: usize,
    ) -> NnResult<Tensor> {
        // Pre-norm architecture: self-attention
        let norm_1 = self.attention_norm.forward(input)?;
        let attn_out = self.attention.forward_with_cache(&norm_1, cache, layer_idx)?;

        let mut x = input.add(&attn_out)?;

        // Cross-attention (if present)
        if let (Some(cross_attn), Some(cross_norm), Some(enc_out)) =
            (&self.cross_attention, &self.cross_attention_norm, encoder_output)
        {
            let norm_cross = cross_norm.forward(&x)?;
            let cross_out = cross_attn.forward(&norm_cross, enc_out)?;
            x = x.add(&cross_out)?;
        }

        // FFN
        let norm_2 = self.ffn_norm.forward(&x)?;
        let ffn_out = self.feed_forward.forward(&norm_2)?;

        x.add(&ffn_out).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_block_basic() {
        let block = TransformerBlock::new(
            64, 4, None, 256, true, 1e-5, true,
            PositionEncoding::None, 128, 10000.0,
        )
        .unwrap();
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_transformer_block_with_cache() {
        let block = TransformerBlock::new(
            64, 4, None, 256, false, 1e-5, true,
            PositionEncoding::None, 128, 10000.0,
        )
        .unwrap();

        let mut cache = KVCache::new(1, 128, 16, 4);

        // Prefill with 8 tokens
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = block.forward_with_cache(&x, None, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
        cache.advance(8);

        // Decode 1 token
        let x = Tensor::randn(vec![1, 1, 64]);
        let y = block.forward_with_cache(&x, None, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 64]);
    }

    #[test]
    fn test_transformer_block_param_count() {
        let block = TransformerBlock::new(
            64, 4, None, 256, false, 1e-5, true,
            PositionEncoding::None, 128, 10000.0,
        )
        .unwrap();
        let (total, frozen) = block.parameter_count();
        assert!(total > 0);
        assert_eq!(frozen, 0);
    }
}
