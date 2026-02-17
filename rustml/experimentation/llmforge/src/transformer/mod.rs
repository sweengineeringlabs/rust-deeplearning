use crate::config::PositionEncoding;
use crate::core::tensor::Tensor;
use crate::error::Result;
use crate::nn::{Linear, Layer, LayerNorm, RMSNorm};
use crate::attention::{MultiHeadAttention, CrossAttention, KVCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Gelu,
    Silu,
    Relu,
    SwiGLU,
}

/// Normalization layer: either standard LayerNorm or RMSNorm.
pub enum NormLayer {
    LayerNorm(LayerNorm),
    RMSNorm(RMSNorm),
}

impl NormLayer {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            NormLayer::LayerNorm(ln) => ln.forward(input),
            NormLayer::RMSNorm(rn) => rn.forward(input),
        }
    }

    pub fn parameter_count(&self) -> (usize, usize) {
        match self {
            NormLayer::LayerNorm(ln) => ln.parameter_count(),
            NormLayer::RMSNorm(rn) => rn.parameter_count(),
        }
    }
}

pub struct FeedForward {
    up_proj: Linear,
    down_proj: Linear,
    gate_proj: Option<Linear>,
    hidden_dim: usize,
    activation: Activation,
}

impl FeedForward {
    pub fn new(d_model: usize, hidden_dim: usize, bias: bool) -> Self {
        Self {
            up_proj: Linear::new(d_model, hidden_dim, bias),
            down_proj: Linear::new(hidden_dim, d_model, bias),
            gate_proj: None,
            hidden_dim,
            activation: Activation::Gelu,
        }
    }

    pub fn with_activation(d_model: usize, hidden_dim: usize, bias: bool, activation: Activation) -> Self {
        let gate_proj = if activation == Activation::SwiGLU {
            Some(Linear::new(d_model, hidden_dim, bias))
        } else {
            None
        };
        Self {
            up_proj: Linear::new(d_model, hidden_dim, bias),
            down_proj: Linear::new(hidden_dim, d_model, bias),
            gate_proj,
            hidden_dim,
            activation,
        }
    }

    /// Construct a SwiGLU feedforward from dimensions.
    pub fn swiglu(d_model: usize, hidden_dim: usize, bias: bool) -> Self {
        Self {
            up_proj: Linear::new(d_model, hidden_dim, bias),
            down_proj: Linear::new(hidden_dim, d_model, bias),
            gate_proj: Some(Linear::new(d_model, hidden_dim, bias)),
            hidden_dim,
            activation: Activation::SwiGLU,
        }
    }

    /// Construct from pre-loaded projection layers.
    /// Derives hidden_dim from up_proj weight shape [hidden_dim, d_model].
    pub fn from_weights(up_proj: Linear, down_proj: Linear) -> Self {
        let hidden_dim = up_proj.weight.shape()[0];
        Self { up_proj, down_proj, gate_proj: None, hidden_dim, activation: Activation::Gelu }
    }

    pub fn from_weights_with_activation(up_proj: Linear, down_proj: Linear, activation: Activation) -> Self {
        let hidden_dim = up_proj.weight.shape()[0];
        Self { up_proj, down_proj, gate_proj: None, hidden_dim, activation }
    }

    /// Construct a SwiGLU feedforward from pre-loaded weights.
    pub fn from_weights_swiglu(up_proj: Linear, gate_proj: Linear, down_proj: Linear) -> Self {
        let hidden_dim = up_proj.weight.shape()[0];
        Self { up_proj, down_proj, gate_proj: Some(gate_proj), hidden_dim, activation: Activation::SwiGLU }
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let (mut total, mut frozen) = (0, 0);
        for proj in [&self.up_proj, &self.down_proj] {
            let (t, f) = proj.parameter_count();
            total += t;
            frozen += f;
        }
        if let Some(ref gate) = self.gate_proj {
            let (t, f) = gate.parameter_count();
            total += t;
            frozen += f;
        }
        (total, frozen)
    }

    /// Toggle native Q4_0×Q8_0 integer matmul on all Linear layers.
    pub fn set_native_q4_matmul(&mut self, enabled: bool) {
        self.up_proj.use_native_q4 = enabled;
        self.down_proj.use_native_q4 = enabled;
        if let Some(ref mut gate) = self.gate_proj {
            gate.use_native_q4 = enabled;
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self.activation {
            Activation::SwiGLU => {
                let gate = self.gate_proj.as_ref()
                    .expect("SwiGLU requires gate_proj")
                    .forward(input)?;
                let gate = gate.silu()?;
                let up = self.up_proj.forward(input)?;
                let h = gate.mul(&up)?;
                self.down_proj.forward(&h)
            }
            _ => {
                let h = self.up_proj.forward(input)?;
                let h = match self.activation {
                    Activation::Gelu => h.gelu()?,
                    Activation::Silu => h.silu()?,
                    Activation::Relu => h.relu()?,
                    Activation::SwiGLU => unreachable!(),
                };
                self.down_proj.forward(&h)
            }
        }
    }
}

pub struct TransformerBlock {
    attention: MultiHeadAttention,
    cross_attention: Option<CrossAttention>,
    cross_attention_norm: Option<NormLayer>,
    feed_forward: FeedForward,
    attention_norm: NormLayer,
    ffn_norm: NormLayer,
}

impl TransformerBlock {
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
    ) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(
                d_model, num_heads, num_kv_heads, bias, causal,
                position_encoding, max_seq_len, rope_theta,
            )?,
            cross_attention: None,
            cross_attention_norm: None,
            feed_forward: FeedForward::new(d_model, hidden_dim, bias),
            attention_norm: NormLayer::LayerNorm(LayerNorm::new(vec![d_model], eps)),
            ffn_norm: NormLayer::LayerNorm(LayerNorm::new(vec![d_model], eps)),
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

    /// Construct from pre-loaded components (RMSNorm variant for Llama).
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

    /// Toggle native Q4_0×Q8_0 integer matmul on all Linear layers in this block.
    pub fn set_native_q4_matmul(&mut self, enabled: bool) {
        self.attention.set_native_q4_matmul(enabled);
        self.feed_forward.set_native_q4_matmul(enabled);
    }

    /// Access MHA for cache sizing queries.
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

    pub fn forward_with_cache(
        &self,
        input: &Tensor,
        encoder_output: Option<&Tensor>,
        cache: &mut KVCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
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

        x.add(&ffn_out)
    }
}
