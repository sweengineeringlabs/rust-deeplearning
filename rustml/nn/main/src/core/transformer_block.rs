//! Transformer block: pre-norm architecture with self-attention + optional cross-attention + FFN.

use crate::api::error::NnResult;
use crate::api::types::PositionEncoding;
use crate::core::attention::MultiHeadAttention;
use crate::core::cross_attention::CrossAttention;
use crate::core::feed_forward::FeedForward;
use crate::core::kv_cache::KVCache;
use crate::core::layer_norm::LayerNorm;
use crate::core::moe::MoeLayer;
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

    /// In-place normalization: for RMSNorm, overwrites the tensor in place
    /// (avoids allocating a separate output tensor when the input is uniquely owned).
    /// For LayerNorm, falls back to allocating since no in-place variant exists.
    pub fn forward_inplace(&self, input: &mut Tensor) -> NnResult<()> {
        match self {
            NormLayer::RMSNorm(rn) => rn.forward_inplace(input),
            NormLayer::LayerNorm(ln) => {
                *input = ln.forward(input)?;
                Ok(())
            }
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
///
/// With 4-norm (Gemma 3): x -> norm1 -> attn -> post_attn_norm -> + -> ffn_norm -> ffn -> post_ffn_norm -> +
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub cross_attention: Option<CrossAttention>,
    pub cross_attention_norm: Option<NormLayer>,
    pub feed_forward: FeedForward,
    pub attention_norm: NormLayer,
    pub ffn_norm: NormLayer,
    /// Post-attention norm (Gemma 3 sandwich norm): applied to attention output before residual add.
    pub post_attention_norm: Option<NormLayer>,
    /// Post-FFN norm (Gemma 3 sandwich norm): applied to FFN output before residual add.
    pub post_ffn_norm: Option<NormLayer>,
    /// Parallel residual connections (Falcon): x = input + attn(norm1(input)) + ffn(norm2(input))
    pub parallel_residual: bool,
    /// Optional MoE layer (Mixtral): replaces feed_forward when present
    pub moe: Option<MoeLayer>,
    /// Use in-place operations in forward_with_cache (default true).
    /// When false, uses allocating `forward()` + `add()` for debugging/benchmarking.
    pub use_inplace_ops: bool,
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
            post_attention_norm: None,
            post_ffn_norm: None,
            parallel_residual: false,
            moe: None,
            use_inplace_ops: true,
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
            post_attention_norm: None,
            post_ffn_norm: None,
            parallel_residual: false,
            moe: None,
            use_inplace_ops: true,
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
            post_attention_norm: None,
            post_ffn_norm: None,
            parallel_residual: false,
            moe: None,
            use_inplace_ops: true,
        }
    }

    /// Construct from pre-loaded components with 4 RMSNorms (Gemma 3 sandwich norm).
    ///
    /// Architecture: norm1 → attn → post_attn_norm → +residual → ffn_norm → ffn → post_ffn_norm → +residual
    pub fn from_weights_rms_4norm(
        attention: MultiHeadAttention,
        feed_forward: FeedForward,
        attention_norm: RMSNorm,
        post_attention_norm: RMSNorm,
        ffn_norm: RMSNorm,
        post_ffn_norm: RMSNorm,
    ) -> Self {
        Self {
            attention,
            cross_attention: None,
            cross_attention_norm: None,
            feed_forward,
            attention_norm: NormLayer::RMSNorm(attention_norm),
            ffn_norm: NormLayer::RMSNorm(ffn_norm),
            post_attention_norm: Some(NormLayer::RMSNorm(post_attention_norm)),
            post_ffn_norm: Some(NormLayer::RMSNorm(post_ffn_norm)),
            parallel_residual: false,
            moe: None,
            use_inplace_ops: true,
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
            post_attention_norm: None,
            post_ffn_norm: None,
            parallel_residual: false,
            moe: None,
            use_inplace_ops: true,
        }
    }

    /// Set parallel residual mode (Falcon-style).
    pub fn set_parallel_residual(&mut self, parallel: bool) {
        self.parallel_residual = parallel;
    }

    /// Toggle in-place operations in `forward_with_cache`.
    /// When true (default), uses `clone` + `forward_inplace` + `add_inplace`.
    /// When false, uses allocating `forward()` + `add()`.
    pub fn set_use_inplace_ops(&mut self, enabled: bool) {
        self.use_inplace_ops = enabled;
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

        if let Some(ref post_attn) = self.post_attention_norm {
            let (t, f) = post_attn.parameter_count();
            total += t;
            frozen += f;
        }
        if let Some(ref post_ffn) = self.post_ffn_norm {
            let (t, f) = post_ffn.parameter_count();
            total += t;
            frozen += f;
        }

        if let Some(ref moe) = self.moe {
            let (t, f) = moe.parameter_count();
            total += t;
            frozen += f;
        }

        (total, frozen)
    }

    /// Toggle native Q4 integer matmul on all Linear layers in this block.
    pub fn set_native_q4_matmul(&mut self, enabled: bool) {
        self.attention.set_native_q4_matmul(enabled);
        self.feed_forward.set_native_q4_matmul(enabled);
        if let Some(ref mut moe) = self.moe {
            moe.set_native_q4_matmul(enabled);
        }
    }

    /// Forward pass without KV cache.
    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        if self.parallel_residual {
            // Falcon: x = input + attn(norm1(input)) + ffn(norm2(input))
            let norm_1 = self.attention_norm.forward(input)?;
            let attn_out = self.attention.forward(&norm_1)?;
            let norm_2 = self.ffn_norm.forward(input)?;
            let ffn_out = self.feed_forward.forward(&norm_2)?;
            let x = input.add(&attn_out)?;
            x.add(&ffn_out).map_err(Into::into)
        } else if let Some(ref moe) = self.moe {
            // Mixtral: MoE replaces FFN
            let norm_1 = self.attention_norm.forward(input)?;
            let attn_out = self.attention.forward(&norm_1)?;
            let x = input.add(&attn_out)?;
            let norm_2 = self.ffn_norm.forward(&x)?;
            let moe_out = moe.forward(&norm_2)?;
            x.add(&moe_out).map_err(Into::into)
        } else {
            // Standard pre-norm (with optional sandwich norms for Gemma 3)
            let norm_1 = self.attention_norm.forward(input)?;
            let attn_out = self.attention.forward(&norm_1)?;
            let attn_out = if let Some(ref pan) = self.post_attention_norm {
                pan.forward(&attn_out)?
            } else { attn_out };
            let x = input.add(&attn_out)?;
            let norm_2 = self.ffn_norm.forward(&x)?;
            let ffn_out = self.feed_forward.forward(&norm_2)?;
            let ffn_out = if let Some(ref pfn) = self.post_ffn_norm {
                pfn.forward(&ffn_out)?
            } else { ffn_out };
            x.add(&ffn_out).map_err(Into::into)
        }
    }

    /// Forward pass with KV cache for autoregressive decoding.
    ///
    /// When `use_inplace_ops` is true (default), uses in-place operations for
    /// residual connections and normalization to minimize allocation overhead.
    /// When false, uses allocating paths (for debugging/benchmarking).
    pub fn forward_with_cache(
        &self,
        input: &Tensor,
        encoder_output: Option<&Tensor>,
        cache: &mut KVCache,
        layer_idx: usize,
    ) -> NnResult<Tensor> {
        if !self.use_inplace_ops {
            return self.forward_with_cache_allocating(input, encoder_output, cache, layer_idx);
        }

        if self.parallel_residual {
            // Falcon: x = input + attn(norm1(input)) + ffn(norm2(input))
            let mut norm_1 = input.clone();
            self.attention_norm.forward_inplace(&mut norm_1)?;
            let mut attn_out = self.attention.forward_with_cache(&norm_1, cache, layer_idx)?;
            let mut norm_2 = input.clone();
            self.ffn_norm.forward_inplace(&mut norm_2)?;
            let ffn_out = self.feed_forward.forward(&norm_2)?;
            // attn_out += input (in-place, attn_out is fresh with refcount 1)
            attn_out.add_inplace(input)?;
            // attn_out += ffn_out
            attn_out.add_inplace(&ffn_out)?;
            Ok(attn_out)
        } else if let Some(ref moe) = self.moe {
            // Mixtral: MoE replaces FFN
            let mut norm_1 = input.clone();
            self.attention_norm.forward_inplace(&mut norm_1)?;
            let mut attn_out = self.attention.forward_with_cache(&norm_1, cache, layer_idx)?;
            attn_out.add_inplace(input)?;
            let mut norm_2 = attn_out.clone();
            self.ffn_norm.forward_inplace(&mut norm_2)?;
            let moe_out = moe.forward(&norm_2)?;
            attn_out.add_inplace(&moe_out)?;
            Ok(attn_out)
        } else {
            // Standard pre-norm (with optional sandwich norms for Gemma 3)
            let mut norm_1 = input.clone();
            self.attention_norm.forward_inplace(&mut norm_1)?;
            let mut attn_out = self.attention.forward_with_cache(&norm_1, cache, layer_idx)?;
            if let Some(ref pan) = self.post_attention_norm {
                attn_out = pan.forward(&attn_out)?;
            }

            // x = attn_out + input (in-place on attn_out, which is fresh)
            attn_out.add_inplace(input)?;
            let mut x = attn_out;

            // Cross-attention (if present)
            if let (Some(cross_attn), Some(cross_norm), Some(enc_out)) =
                (&self.cross_attention, &self.cross_attention_norm, encoder_output)
            {
                let norm_cross = cross_norm.forward(&x)?;
                let cross_out = cross_attn.forward(&norm_cross, enc_out)?;
                x.add_inplace(&cross_out)?;
            }

            // FFN
            let mut norm_2 = x.clone();
            self.ffn_norm.forward_inplace(&mut norm_2)?;
            let mut ffn_out = self.feed_forward.forward(&norm_2)?;
            if let Some(ref pfn) = self.post_ffn_norm {
                ffn_out = pfn.forward(&ffn_out)?;
            }

            // x += ffn_out (in-place on x, which is uniquely owned)
            x.add_inplace(&ffn_out)?;
            Ok(x)
        }
    }

    /// Allocating variant of forward_with_cache (baseline path).
    fn forward_with_cache_allocating(
        &self,
        input: &Tensor,
        encoder_output: Option<&Tensor>,
        cache: &mut KVCache,
        layer_idx: usize,
    ) -> NnResult<Tensor> {
        if self.parallel_residual {
            let norm_1 = self.attention_norm.forward(input)?;
            let attn_out = self.attention.forward_with_cache(&norm_1, cache, layer_idx)?;
            let norm_2 = self.ffn_norm.forward(input)?;
            let ffn_out = self.feed_forward.forward(&norm_2)?;
            let x = input.add(&attn_out)?;
            x.add(&ffn_out).map_err(Into::into)
        } else if let Some(ref moe) = self.moe {
            let norm_1 = self.attention_norm.forward(input)?;
            let attn_out = self.attention.forward_with_cache(&norm_1, cache, layer_idx)?;
            let x = input.add(&attn_out)?;
            let norm_2 = self.ffn_norm.forward(&x)?;
            let moe_out = moe.forward(&norm_2)?;
            x.add(&moe_out).map_err(Into::into)
        } else {
            let norm_1 = self.attention_norm.forward(input)?;
            let attn_out = self.attention.forward_with_cache(&norm_1, cache, layer_idx)?;
            let attn_out = if let Some(ref pan) = self.post_attention_norm {
                pan.forward(&attn_out)?
            } else { attn_out };
            let x = input.add(&attn_out)?;

            if let (Some(cross_attn), Some(cross_norm), Some(enc_out)) =
                (&self.cross_attention, &self.cross_attention_norm, encoder_output)
            {
                let norm_cross = cross_norm.forward(&x)?;
                let cross_out = cross_attn.forward(&norm_cross, enc_out)?;
                let x = x.add(&cross_out)?;
                let norm_2 = self.ffn_norm.forward(&x)?;
                let ffn_out = self.feed_forward.forward(&norm_2)?;
                let ffn_out = if let Some(ref pfn) = self.post_ffn_norm {
                    pfn.forward(&ffn_out)?
                } else { ffn_out };
                x.add(&ffn_out).map_err(Into::into)
            } else {
                let norm_2 = self.ffn_norm.forward(&x)?;
                let ffn_out = self.feed_forward.forward(&norm_2)?;
                let ffn_out = if let Some(ref pfn) = self.post_ffn_norm {
                    pfn.forward(&ffn_out)?
                } else { ffn_out };
                x.add(&ffn_out).map_err(Into::into)
            }
        }
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
    fn test_transformer_block_parallel_residual() {
        let mut block = TransformerBlock::new(
            64, 4, None, 256, true, 1e-5, true,
            PositionEncoding::None, 128, 10000.0,
        )
        .unwrap();
        block.set_parallel_residual(true);
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_transformer_block_parallel_residual_with_cache() {
        let mut block = TransformerBlock::new(
            64, 4, None, 256, false, 1e-5, true,
            PositionEncoding::None, 128, 10000.0,
        )
        .unwrap();
        block.set_parallel_residual(true);

        let mut cache = KVCache::new(1, 128, 16, 4);

        let x = Tensor::randn(vec![1, 8, 64]);
        let y = block.forward_with_cache(&x, None, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
        cache.advance(8);

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

    #[test]
    fn test_transformer_block_4norm() {
        let attention = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::RoPE, 128, 10000.0,
        ).unwrap();
        let feed_forward = FeedForward::new(64, 256, false);
        let block = TransformerBlock::from_weights_rms_4norm(
            attention,
            feed_forward,
            RMSNorm::new(64, 1e-6),
            RMSNorm::new(64, 1e-6),
            RMSNorm::new(64, 1e-6),
            RMSNorm::new(64, 1e-6),
        );

        // Forward without cache
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);

        // Forward with cache
        let mut cache = KVCache::new(1, 128, 16, 4);
        let y = block.forward_with_cache(&x, None, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
    }

    // ==================== NormLayer::forward_inplace tests ====================

    #[test]
    fn test_norm_layer_inplace_rms_matches_forward() {
        let norm = NormLayer::RMSNorm(RMSNorm::new(64, 1e-6));
        let x = Tensor::randn(vec![1, 4, 64]);

        let y_alloc = norm.forward(&x).unwrap();

        let mut y_inplace = x.clone();
        norm.forward_inplace(&mut y_inplace).unwrap();

        let d_alloc = y_alloc.as_slice_f32().unwrap();
        let d_inplace = y_inplace.as_slice_f32().unwrap();
        assert_eq!(d_alloc.len(), d_inplace.len());
        for i in 0..d_alloc.len() {
            assert!((d_alloc[i] - d_inplace[i]).abs() < 1e-5,
                "rms norm_layer inplace mismatch at index {}: {} vs {}", i, d_alloc[i], d_inplace[i]);
        }
    }

    #[test]
    fn test_norm_layer_inplace_layernorm_matches_forward() {
        let norm = NormLayer::LayerNorm(LayerNorm::with_eps(64, 1e-5));
        let x = Tensor::randn(vec![1, 4, 64]);

        let y_alloc = norm.forward(&x).unwrap();

        let mut y_inplace = x.clone();
        norm.forward_inplace(&mut y_inplace).unwrap();

        let d_alloc = y_alloc.as_slice_f32().unwrap();
        let d_inplace = y_inplace.as_slice_f32().unwrap();
        assert_eq!(d_alloc.len(), d_inplace.len());
        for i in 0..d_alloc.len() {
            assert!((d_alloc[i] - d_inplace[i]).abs() < 1e-5,
                "layernorm norm_layer inplace mismatch at {}: {} vs {}", i, d_alloc[i], d_inplace[i]);
        }
    }

    #[test]
    fn test_transformer_block_forward_with_cache_rms_inplace() {
        // Build a block with RMSNorm (exercises the inplace path in forward_with_cache)
        let block = TransformerBlock::from_weights_rms(
            MultiHeadAttention::new(
                64, 4, None, false, true,
                PositionEncoding::RoPE, 128, 10000.0,
            ).unwrap(),
            FeedForward::new(64, 256, false),
            RMSNorm::new(64, 1e-6),
            RMSNorm::new(64, 1e-6),
        );

        let mut cache = KVCache::new(1, 128, 16, 4);

        // Prefill
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = block.forward_with_cache(&x, None, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
        cache.advance(8);

        // Decode step — this is the hot path exercising inplace norms
        let x2 = Tensor::randn(vec![1, 1, 64]);
        let y2 = block.forward_with_cache(&x2, None, &mut cache, 0).unwrap();
        assert_eq!(y2.shape(), &[1, 1, 64]);
        // Ensure output is finite
        let flat = y2.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "decode output contains NaN/Inf");
    }

    #[test]
    fn test_transformer_block_forward_with_cache_layernorm_inplace() {
        // Build a block with LayerNorm (NormLayer::forward_inplace falls back to allocating)
        let block = TransformerBlock::new(
            64, 4, None, 256, true, 1e-5, true,
            PositionEncoding::None, 128, 10000.0,
        ).unwrap();

        let mut cache = KVCache::new(1, 128, 16, 4);

        let x = Tensor::randn(vec![1, 6, 64]);
        let y = block.forward_with_cache(&x, None, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 6, 64]);
        cache.advance(6);

        let x2 = Tensor::randn(vec![1, 1, 64]);
        let y2 = block.forward_with_cache(&x2, None, &mut cache, 0).unwrap();
        assert_eq!(y2.shape(), &[1, 1, 64]);
        let flat = y2.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "decode output contains NaN/Inf");
    }

    #[test]
    fn test_transformer_block_4norm_inplace_with_cache() {
        // 4-norm (Gemma 3) block with RMSNorm — exercises inplace for attention_norm + ffn_norm
        let attention = MultiHeadAttention::new(
            64, 4, None, false, true,
            PositionEncoding::RoPE, 128, 10000.0,
        ).unwrap();
        let feed_forward = FeedForward::new(64, 256, false);
        let block = TransformerBlock::from_weights_rms_4norm(
            attention,
            feed_forward,
            RMSNorm::new(64, 1e-6),
            RMSNorm::new(64, 1e-6),
            RMSNorm::new(64, 1e-6),
            RMSNorm::new(64, 1e-6),
        );

        let mut cache = KVCache::new(1, 128, 16, 4);

        // Prefill
        let x = Tensor::randn(vec![1, 4, 64]);
        let y = block.forward_with_cache(&x, None, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 4, 64]);
        cache.advance(4);

        // Decode
        let x2 = Tensor::randn(vec![1, 1, 64]);
        let y2 = block.forward_with_cache(&x2, None, &mut cache, 0).unwrap();
        assert_eq!(y2.shape(), &[1, 1, 64]);
        let flat = y2.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    // ==================== Configurable inplace ops tests ====================

    #[test]
    fn test_transformer_block_inplace_vs_allocating_with_cache() {
        // Build two identical blocks, one with inplace ops, one without
        let mut block_inplace = TransformerBlock::new(
            64, 4, None, 256, false, 1e-5, true,
            PositionEncoding::None, 128, 10000.0,
        ).unwrap();
        block_inplace.set_use_inplace_ops(true);

        let mut block_alloc = TransformerBlock::new(
            64, 4, None, 256, false, 1e-5, true,
            PositionEncoding::None, 128, 10000.0,
        ).unwrap();
        block_alloc.set_use_inplace_ops(false);

        // Copy weights from inplace to allocating block so they match
        // (new() creates random weights, so we need identical blocks)
        // Instead: use the same block and toggle the flag
        let mut block = TransformerBlock::new(
            64, 4, None, 256, false, 1e-5, true,
            PositionEncoding::None, 128, 10000.0,
        ).unwrap();

        let x = Tensor::randn(vec![1, 4, 64]);

        // Run with inplace
        block.set_use_inplace_ops(true);
        let mut cache1 = KVCache::new(1, 128, 16, 4);
        let y_inplace = block.forward_with_cache(&x, None, &mut cache1, 0).unwrap();

        // Run with allocating
        block.set_use_inplace_ops(false);
        let mut cache2 = KVCache::new(1, 128, 16, 4);
        let y_alloc = block.forward_with_cache(&x, None, &mut cache2, 0).unwrap();

        let d1 = y_inplace.as_slice_f32().unwrap();
        let d2 = y_alloc.as_slice_f32().unwrap();
        assert_eq!(d1.len(), d2.len());
        for i in 0..d1.len() {
            assert!((d1[i] - d2[i]).abs() < 1e-4,
                "inplace vs alloc block mismatch at {}: {} vs {}", i, d1[i], d2[i]);
        }
    }

    #[test]
    fn test_transformer_block_allocating_forward_with_cache_basic() {
        // Ensure the allocating path produces finite output
        let mut block = TransformerBlock::new(
            64, 4, None, 256, false, 1e-5, true,
            PositionEncoding::None, 128, 10000.0,
        ).unwrap();
        block.set_use_inplace_ops(false);

        let mut cache = KVCache::new(1, 128, 16, 4);
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = block.forward_with_cache(&x, None, &mut cache, 0).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
        let flat = y.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }
}
