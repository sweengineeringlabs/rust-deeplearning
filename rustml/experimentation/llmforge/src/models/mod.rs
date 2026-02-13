use std::collections::HashMap;
use crate::core::tensor::{Tensor, DType};
use crate::error::{LLMForgeError, Result};
use crate::nn::{Embedding, Linear, Layer, LayerNorm};
use crate::transformer::{TransformerBlock, FeedForward};
use crate::attention::{MultiHeadAttention, KVCache};

pub struct LlmModel {
    pub token_embedding: Embedding,
    pub pos_embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: LayerNorm,
    pub output: Linear,
    pub d_model: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

impl LlmModel {
    pub fn new(config: &crate::config::ModelConfig) -> Result<Self> {
        let max_seq_len = config.max_seq_len;
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let hidden_dim = config.hidden_dim;
        let num_layers = config.n_layers;
        let vocab_size = config.vocab_size;
        let bias = config.use_bias.unwrap_or(false);
        let eps = config.norm_eps;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new(
                d_model,
                num_heads,
                hidden_dim,
                bias,
                eps,
            )?);
        }

        Ok(Self {
            token_embedding: Embedding::new(vocab_size, d_model),
            pos_embedding: Embedding::new(max_seq_len, d_model),
            layers,
            norm: LayerNorm::new(vec![d_model], eps),
            output: Linear::new(d_model, vocab_size, false),
            d_model,
            vocab_size,
            max_seq_len,
            n_layers: num_layers,
            n_heads: num_heads,
        })
    }

    /// Construct model from pre-loaded and remapped weights (internal names).
    ///
    /// Expects weights already remapped via `WeightMap::remap()`.
    /// Positional embedding is random-initialized (placeholder until RoPE).
    pub fn from_pretrained(config: &crate::config::ModelConfig, weights: HashMap<String, Tensor>) -> Result<Self> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let vocab_size = config.vocab_size;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;

        let get_tensor = |key: &str| -> Result<Tensor> {
            weights.get(key)
                .ok_or_else(|| LLMForgeError::NotImplemented(
                    format!("Missing weight: {}", key)
                ))
                .and_then(|t| t.to_f32())
        };

        // Token embedding
        let token_embedding = Embedding::from_weights(get_tensor("token_embedding.weight")?);

        // Positional embedding: random init (placeholder until RoPE)
        let pos_embedding = Embedding::new(max_seq_len, d_model);

        // Transformer layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let q_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.attention.q_proj.weight", i))?,
                None,
            );
            let k_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.attention.k_proj.weight", i))?,
                None,
            );
            let v_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.attention.v_proj.weight", i))?,
                None,
            );
            let out_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.attention.out_proj.weight", i))?,
                None,
            );
            let attention = MultiHeadAttention::from_weights(d_model, num_heads, q_proj, k_proj, v_proj, out_proj)?;

            let up_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.feed_forward.up_proj.weight", i))?,
                None,
            );
            let down_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.feed_forward.down_proj.weight", i))?,
                None,
            );
            let feed_forward = FeedForward::from_weights(up_proj, down_proj);

            let attention_norm = LayerNorm::from_weight_only(
                get_tensor(&format!("layers.{}.attention_norm.weight", i))?,
                eps,
            );
            let ffn_norm = LayerNorm::from_weight_only(
                get_tensor(&format!("layers.{}.ffn_norm.weight", i))?,
                eps,
            );

            layers.push(TransformerBlock::from_weights(attention, feed_forward, attention_norm, ffn_norm));
        }

        // Final norm
        let norm = LayerNorm::from_weight_only(get_tensor("norm.weight")?, eps);

        // Output projection
        let output = Linear::from_weights(get_tensor("output.weight")?, None);

        Ok(Self {
            token_embedding,
            pos_embedding,
            layers,
            norm,
            output,
            d_model,
            vocab_size,
            max_seq_len,
            n_layers: num_layers,
            n_heads: num_heads,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // [Batch, Seq] -> [Batch, Seq, D_Model]
        let x_emb = self.token_embedding.forward(input_ids)?;

        // Positional Embeddings
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        if seq_len > self.max_seq_len {
             return Err(crate::error::LLMForgeError::IndexOutOfBounds{ index: seq_len, dim: 1, size: self.max_seq_len });
        }

        // Generate pos indices [Batch, Seq]
        let mut pos_data = Vec::with_capacity(batch_size * seq_len);
        for _ in 0..batch_size {
            for i in 0..seq_len {
                pos_data.push(i as f32);
            }
        }

        let pos_bytes = crate::core::tensor::f32_vec_to_bytes(pos_data);
        let pos_ids = Tensor::new(pos_bytes, vec![batch_size, seq_len], DType::F32);

        let p_emb = self.pos_embedding.forward(&pos_ids)?;

        let mut x = x_emb.add(&p_emb)?;

        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Final normalization
        let x = self.norm.forward(&x)?;

        // Output projection
        self.output.forward(&x)
    }

    pub fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KVCache) -> Result<Tensor> {
        // [Batch, Seq] -> [Batch, Seq, D_Model]
        let x_emb = self.token_embedding.forward(input_ids)?;

        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];
        let start_pos = cache.current_len;

        if start_pos + seq_len > self.max_seq_len {
             return Err(crate::error::LLMForgeError::SequenceLengthExceeded {
                 max: self.max_seq_len,
                 actual: start_pos + seq_len,
             });
        }

        // Generate pos indices [Batch, Seq] starting from start_pos
        let mut pos_data = Vec::with_capacity(batch_size * seq_len);
        for _ in 0..batch_size {
            for i in 0..seq_len {
                pos_data.push((start_pos + i) as f32);
            }
        }

        let pos_bytes = crate::core::tensor::f32_vec_to_bytes(pos_data);
        let pos_ids = Tensor::new(pos_bytes, vec![batch_size, seq_len], DType::F32);

        let p_emb = self.pos_embedding.forward(&pos_ids)?;
        let mut x = x_emb.add(&p_emb)?;

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_with_cache(&x, cache, i)?;
        }

        let x = self.norm.forward(&x)?;
        self.output.forward(&x)
    }
}
