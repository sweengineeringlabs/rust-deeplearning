//! Attention mechanisms including Causal Self-Attention for GPT

use crate::core::linear::Linear;
use crate::api::error::NnResult;
use rustml_core::Tensor;

/// Multi-head attention configuration
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionConfig {
    /// Model dimension (embedding size)
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dropout rate (not implemented yet)
    pub dropout: f32,
    /// Whether to use bias in projections
    pub bias: bool,
}

impl Default for MultiHeadAttentionConfig {
    fn default() -> Self {
        Self {
            d_model: 768,
            n_heads: 12,
            dropout: 0.0,
            bias: true,
        }
    }
}

/// Multi-head attention layer
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Query projection
    pub wq: Linear,
    /// Key projection
    pub wk: Linear,
    /// Value projection
    pub wv: Linear,
    /// Output projection
    pub wo: Linear,
    /// Configuration
    pub config: MultiHeadAttentionConfig,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(config: MultiHeadAttentionConfig) -> Self {
        let d_model = config.d_model;
        let wq = Linear::new(d_model, d_model);
        let wk = Linear::new(d_model, d_model);
        let wv = Linear::new(d_model, d_model);
        let wo = Linear::new(d_model, d_model);

        Self { wq, wk, wv, wo, config }
    }

    /// Forward pass for encoder-style attention (no causal mask)
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        self.forward_with_mask(x, None)
    }

    /// Forward with optional attention mask
    pub fn forward_with_mask(&self, x: &Tensor, mask: Option<&Tensor>) -> NnResult<Tensor> {
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];
        let n_heads = self.config.n_heads;
        let head_dim = d_model / n_heads;

        // Project to Q, K, V
        let q = self.wq.forward(x)?; // [B, T, D]
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        // Reshape to [B, H, T, D/H]
        let q = q
            .reshape(vec![batch_size, seq_len, n_heads, head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape(vec![batch_size, seq_len, n_heads, head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape(vec![batch_size, seq_len, n_heads, head_dim])?
            .transpose(1, 2)?;

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(&k.t()?)?.div_scalar(scale);

        // Apply mask if provided
        let scores = if let Some(m) = mask {
            scores.masked_fill(m, f32::NEG_INFINITY)?
        } else {
            scores
        };

        // Softmax
        let attn = scores.softmax(-1)?;

        // attn @ V
        let out = attn.matmul(&v)?; // [B, H, T, D/H]

        // Reshape back to [B, T, D]
        let out = out
            .transpose(1, 2)?
            .reshape(vec![batch_size, seq_len, d_model])?;

        // Output projection
        self.wo.forward(&out)
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
            return Err(crate::api::error::NnError::ShapeMismatch(format!(
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
            .reshape(vec![batch_size, seq_len, self.n_head, head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape(vec![batch_size, seq_len, self.n_head, head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape(vec![batch_size, seq_len, self.n_head, head_dim])?
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
            .reshape(vec![batch_size, seq_len, n_embd])?;

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
    fn test_multi_head_attention() {
        let config = MultiHeadAttentionConfig {
            d_model: 64,
            n_heads: 4,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(config);
        let x = Tensor::randn(vec![2, 8, 64]);
        let y = mha.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8, 64]);
    }
}
