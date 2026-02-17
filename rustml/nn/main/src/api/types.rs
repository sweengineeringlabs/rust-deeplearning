//! Types for neural network layers

/// Position encoding strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionEncoding {
    /// Learned position embeddings (GPT-2 style)
    Learned,
    /// Rotary Position Encoding (Llama style)
    #[serde(rename = "rope")]
    RoPE,
    /// Attention with Linear Biases
    #[serde(rename = "alibi")]
    ALiBi,
    /// No position encoding
    None,
}

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Gelu,
    Silu,
    Relu,
    SwiGLU,
    GeGLU,
}
