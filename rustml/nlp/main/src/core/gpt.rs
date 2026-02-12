//! GPT-2 Model Implementation
//!
//! This module provides a complete GPT-2 implementation compatible with
//! HuggingFace pre-trained weights.

use crate::api::error::{NlpError, NlpResult};
use crate::api::types::GptConfig;
use rustml_core::Tensor;
use rustml_hub::{Gpt2WeightMapper, WeightMapper};
use rustml_nn::{CausalSelfAttention, Embedding, LayerNorm, Linear};
use std::collections::HashMap;

/// GPT-2 MLP (Feed-Forward Network)
///
/// Structure: fc -> GELU -> proj
/// - fc: Linear(n_embd, 4 * n_embd)
/// - proj: Linear(4 * n_embd, n_embd)
#[derive(Debug, Clone)]
pub struct GptMlp {
    /// First linear layer (expansion)
    pub c_fc: Linear,
    /// Second linear layer (projection)
    pub c_proj: Linear,
}

impl GptMlp {
    /// Create a new MLP
    pub fn new(n_embd: usize) -> Self {
        let hidden_dim = 4 * n_embd;
        Self {
            c_fc: Linear::new(n_embd, hidden_dim),
            c_proj: Linear::new(hidden_dim, n_embd),
        }
    }

    /// Load from weights
    pub fn from_weights(
        c_fc_weight: Tensor,
        c_fc_bias: Option<Tensor>,
        c_proj_weight: Tensor,
        c_proj_bias: Option<Tensor>,
    ) -> NlpResult<Self> {
        Ok(Self {
            c_fc: Linear::from_weights(c_fc_weight, c_fc_bias)?,
            c_proj: Linear::from_weights(c_proj_weight, c_proj_bias)?,
        })
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> NlpResult<Tensor> {
        let h = self.c_fc.forward(x)?;
        let h = h.gelu();
        let out = self.c_proj.forward(&h)?;
        Ok(out)
    }
}

/// GPT-2 Transformer Block
///
/// Structure:
/// - LN -> Attention -> Residual
/// - LN -> MLP -> Residual
#[derive(Debug, Clone)]
pub struct GptBlock {
    /// Pre-attention layer norm
    pub ln_1: LayerNorm,
    /// Causal self-attention
    pub attn: CausalSelfAttention,
    /// Pre-MLP layer norm
    pub ln_2: LayerNorm,
    /// Feed-forward network
    pub mlp: GptMlp,
}

impl GptBlock {
    /// Create a new transformer block
    pub fn new(config: &GptConfig) -> Self {
        Self {
            ln_1: LayerNorm::with_eps(config.n_embd, config.layer_norm_eps),
            attn: CausalSelfAttention::new(config.n_embd, config.n_head),
            ln_2: LayerNorm::with_eps(config.n_embd, config.layer_norm_eps),
            mlp: GptMlp::new(config.n_embd),
        }
    }

    /// Load from weights
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        config: &GptConfig,
    ) -> NlpResult<Self> {
        let get_weight = |name: &str| -> NlpResult<Tensor> {
            weights
                .get(&format!("{}.{}", prefix, name))
                .cloned()
                .ok_or_else(|| NlpError::ModelError(format!("Missing weight: {}.{}", prefix, name)))
        };

        let get_weight_opt = |name: &str| -> Option<Tensor> {
            weights.get(&format!("{}.{}", prefix, name)).cloned()
        };

        let ln_1 = LayerNorm::from_weights(
            get_weight("ln_1.weight")?,
            get_weight("ln_1.bias")?,
            config.layer_norm_eps,
        )?;

        let attn = CausalSelfAttention::from_weights(
            get_weight("attn.c_attn.weight")?,
            get_weight_opt("attn.c_attn.bias"),
            get_weight("attn.c_proj.weight")?,
            get_weight_opt("attn.c_proj.bias"),
            config.n_head,
        )?;

        let ln_2 = LayerNorm::from_weights(
            get_weight("ln_2.weight")?,
            get_weight("ln_2.bias")?,
            config.layer_norm_eps,
        )?;

        let mlp = GptMlp::from_weights(
            get_weight("mlp.c_fc.weight")?,
            get_weight_opt("mlp.c_fc.bias"),
            get_weight("mlp.c_proj.weight")?,
            get_weight_opt("mlp.c_proj.bias"),
        )?;

        Ok(Self { ln_1, attn, ln_2, mlp })
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> NlpResult<Tensor> {
        // Attention with residual
        let h = self.ln_1.forward(x)?;
        let attn_out = self.attn.forward(&h)?;
        let x = x.add(&attn_out)?;

        // MLP with residual
        let h = self.ln_2.forward(&x)?;
        let mlp_out = self.mlp.forward(&h)?;
        let x = x.add(&mlp_out)?;

        Ok(x)
    }
}

/// GPT-2 Model
///
/// Full GPT-2 architecture:
/// - Token embeddings (wte)
/// - Position embeddings (wpe)
/// - Transformer blocks
/// - Final layer norm (ln_f)
///
/// Output: logits over vocabulary
#[derive(Debug, Clone)]
pub struct GptModel {
    /// Model configuration
    pub config: GptConfig,
    /// Token embeddings
    pub wte: Embedding,
    /// Position embeddings
    pub wpe: Embedding,
    /// Transformer blocks
    pub blocks: Vec<GptBlock>,
    /// Final layer normalization
    pub ln_f: LayerNorm,
}

impl GptModel {
    /// Create a new randomly initialized GPT model
    pub fn new(config: GptConfig) -> Self {
        let wte = Embedding::new(config.vocab_size, config.n_embd);
        let wpe = Embedding::new(config.n_positions, config.n_embd);
        let blocks: Vec<GptBlock> = (0..config.n_layer)
            .map(|_| GptBlock::new(&config))
            .collect();
        let ln_f = LayerNorm::with_eps(config.n_embd, config.layer_norm_eps);

        Self {
            config,
            wte,
            wpe,
            blocks,
            ln_f,
        }
    }

    /// Load model from HuggingFace Hub weights
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `weights` - Raw weights from HuggingFace (before mapping)
    pub fn from_hub_weights(
        config: GptConfig,
        weights: HashMap<String, Tensor>,
    ) -> NlpResult<Self> {
        // Map HuggingFace weight names to our format
        let mapper = Gpt2WeightMapper::new(config.n_layer);
        let weights = mapper.map_weights(weights)?;

        Self::from_weights(config, weights)
    }

    /// Load model from pre-mapped weights
    pub fn from_weights(config: GptConfig, weights: HashMap<String, Tensor>) -> NlpResult<Self> {
        let get_weight = |name: &str| -> NlpResult<Tensor> {
            weights
                .get(name)
                .cloned()
                .ok_or_else(|| NlpError::ModelError(format!("Missing weight: {}", name)))
        };

        // Load embeddings
        let wte = Embedding::from_weights(get_weight("wte.weight")?)?;
        let wpe = Embedding::from_weights(get_weight("wpe.weight")?)?;

        // Load transformer blocks
        let blocks: Result<Vec<GptBlock>, _> = (0..config.n_layer)
            .map(|i| GptBlock::from_weights(&weights, &format!("blocks.{}", i), &config))
            .collect();
        let blocks = blocks?;

        // Load final layer norm
        let ln_f = LayerNorm::from_weights(
            get_weight("ln_f.weight")?,
            get_weight("ln_f.bias")?,
            config.layer_norm_eps,
        )?;

        Ok(Self {
            config,
            wte,
            wpe,
            blocks,
            ln_f,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape [batch_size, seq_len]
    ///
    /// # Returns
    /// Logits over vocabulary, shape [batch_size, seq_len, vocab_size]
    pub fn forward(&self, input_ids: &Tensor) -> NlpResult<Tensor> {
        let shape = input_ids.shape();
        let seq_len = shape[shape.len() - 1];

        // Check sequence length
        if seq_len > self.config.n_positions {
            return Err(NlpError::ModelError(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.config.n_positions
            )));
        }

        // Create position IDs: [0, 1, 2, ..., seq_len-1]
        let position_ids = Tensor::arange(0.0, seq_len as f32, 1.0)?;
        // Broadcast to match input shape
        let position_ids = if shape.len() == 2 {
            position_ids.unsqueeze(0)?.broadcast_to(&input_ids.shape().into())?
        } else {
            position_ids
        };

        // Get embeddings
        let token_embeds = self.wte.forward(input_ids)?;
        let position_embeds = self.wpe.forward(&position_ids)?;

        // Combine embeddings
        let mut hidden_states = token_embeds.add(&position_embeds)?;

        // Pass through transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }

        // Final layer norm
        hidden_states = self.ln_f.forward(&hidden_states)?;

        // Project to vocabulary (weight tying: use wte.weight.T)
        let logits = hidden_states.matmul(&self.wte.weight.t()?)?;

        Ok(logits)
    }

    /// Get the model's vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get the model's embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.config.n_embd
    }

    /// Get the model's maximum sequence length
    pub fn max_sequence_length(&self) -> usize {
        self.config.n_positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt_config_presets() {
        let small = GptConfig::gpt2_small();
        assert_eq!(small.n_embd, 768);
        assert_eq!(small.n_layer, 12);
        assert_eq!(small.n_head, 12);

        let medium = GptConfig::gpt2_medium();
        assert_eq!(medium.n_embd, 1024);
        assert_eq!(medium.n_layer, 24);

        let large = GptConfig::gpt2_large();
        assert_eq!(large.n_embd, 1280);
        assert_eq!(large.n_layer, 36);

        let xl = GptConfig::gpt2_xl();
        assert_eq!(xl.n_embd, 1600);
        assert_eq!(xl.n_layer, 48);
    }

    #[test]
    fn test_gpt_model_creation() {
        // Create a tiny model for testing
        let config = GptConfig {
            vocab_size: 100,
            n_positions: 32,
            n_embd: 64,
            n_layer: 2,
            n_head: 4,
            layer_norm_eps: 1e-5,
        };

        let model = GptModel::new(config.clone());
        assert_eq!(model.blocks.len(), 2);
        assert_eq!(model.vocab_size(), 100);
    }

    #[test]
    fn test_gpt_forward_shape() {
        let config = GptConfig {
            vocab_size: 100,
            n_positions: 32,
            n_embd: 64,
            n_layer: 2,
            n_head: 4,
            layer_norm_eps: 1e-5,
        };

        let model = GptModel::new(config);

        // Input: [batch=2, seq=8]
        let input_ids = Tensor::from_vec(
            (0..16).map(|i| (i % 100) as f32).collect(),
            vec![2, 8],
        )
        .unwrap();

        let logits = model.forward(&input_ids).unwrap();
        assert_eq!(logits.shape(), &[2, 8, 100]);
    }

    #[test]
    fn test_gpt_mlp() {
        let mlp = GptMlp::new(64);
        let x = Tensor::randn(vec![2, 8, 64]);
        let y = mlp.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8, 64]);
    }
}
