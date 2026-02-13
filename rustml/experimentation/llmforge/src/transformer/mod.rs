use crate::core::tensor::Tensor;
use crate::error::Result;
use crate::nn::{Linear, Layer, LayerNorm};
use crate::attention::{MultiHeadAttention, KVCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Gelu,
    Silu,
    Relu,
}

pub struct FeedForward {
    up_proj: Linear,
    down_proj: Linear,
    hidden_dim: usize,
    activation: Activation,
}

impl FeedForward {
    pub fn new(d_model: usize, hidden_dim: usize, bias: bool) -> Self {
        Self {
            up_proj: Linear::new(d_model, hidden_dim, bias),
            down_proj: Linear::new(hidden_dim, d_model, bias),
            hidden_dim,
            activation: Activation::Gelu,
        }
    }

    pub fn with_activation(d_model: usize, hidden_dim: usize, bias: bool, activation: Activation) -> Self {
        Self {
            up_proj: Linear::new(d_model, hidden_dim, bias),
            down_proj: Linear::new(hidden_dim, d_model, bias),
            hidden_dim,
            activation,
        }
    }

    /// Construct from pre-loaded projection layers.
    /// Derives hidden_dim from up_proj weight shape [hidden_dim, d_model].
    pub fn from_weights(up_proj: Linear, down_proj: Linear) -> Self {
        let hidden_dim = up_proj.weight.shape()[0];
        Self { up_proj, down_proj, hidden_dim, activation: Activation::Gelu }
    }

    pub fn from_weights_with_activation(up_proj: Linear, down_proj: Linear, activation: Activation) -> Self {
        let hidden_dim = up_proj.weight.shape()[0];
        Self { up_proj, down_proj, hidden_dim, activation }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let h = self.up_proj.forward(input)?;
        let h = match self.activation {
            Activation::Gelu => h.gelu()?,
            Activation::Silu => h.silu()?,
            Activation::Relu => h.relu()?,
        };
        self.down_proj.forward(&h)
    }
}

pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    attention_norm: LayerNorm,
    ffn_norm: LayerNorm,
}

impl TransformerBlock {
    pub fn new(d_model: usize, num_heads: usize, hidden_dim: usize, bias: bool, eps: f32) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(d_model, num_heads, bias)?,
            feed_forward: FeedForward::new(d_model, hidden_dim, bias),
            attention_norm: LayerNorm::new(vec![d_model], eps),
            ffn_norm: LayerNorm::new(vec![d_model], eps),
        })
    }

    /// Construct from pre-loaded components.
    pub fn from_weights(
        attention: MultiHeadAttention,
        feed_forward: FeedForward,
        attention_norm: LayerNorm,
        ffn_norm: LayerNorm,
    ) -> Self {
        Self { attention, feed_forward, attention_norm, ffn_norm }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Pre-norm architecture
        // x = x + attn(ln(x))
        let norm_1 = self.attention_norm.forward(input)?;
        let attn_out = self.attention.forward(&norm_1)?;
        let x = input.add(&attn_out)?;

        // x = x + ffn(ln(x))
        let norm_2 = self.ffn_norm.forward(&x)?;
        let ffn_out = self.feed_forward.forward(&norm_2)?;
        x.add(&ffn_out)
    }

    pub fn forward_with_cache(&self, input: &Tensor, cache: &mut KVCache, layer_idx: usize) -> Result<Tensor> {
        // Pre-norm architecture
        let norm_1 = self.attention_norm.forward(input)?;
        let attn_out = self.attention.forward_with_cache(&norm_1, cache, layer_idx)?;
        let x = input.add(&attn_out)?;

        let norm_2 = self.ffn_norm.forward(&x)?;
        let ffn_out = self.feed_forward.forward(&norm_2)?;
        x.add(&ffn_out)
    }
}
