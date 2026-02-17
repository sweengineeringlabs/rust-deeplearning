use std::collections::HashMap;
use crate::config::PositionEncoding;
use crate::core::tensor::{Tensor, DType};
use crate::error::{LLMForgeError, Result};
use crate::nn::{Embedding, Linear, Layer, LayerNorm, RMSNorm, Freezable};
use crate::transformer::{TransformerBlock, NormLayer, FeedForward, Activation};
use crate::attention::{MultiHeadAttention, KVCache};

pub struct LlmModel {
    pub token_embedding: Embedding,
    pub pos_embedding: Option<Embedding>,
    pub layers: Vec<TransformerBlock>,
    pub norm: NormLayer,
    pub output: Linear,
    pub d_model: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub position_encoding: PositionEncoding,
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
        let n_kv_heads = config.n_kv_heads.unwrap_or(num_heads);
        let position_encoding = config.position_encoding;
        let causal = config.causal;
        let rope_theta = config.rope_theta;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new(
                d_model,
                num_heads,
                config.n_kv_heads,
                hidden_dim,
                bias,
                eps,
                causal,
                position_encoding,
                max_seq_len,
                rope_theta,
            )?);
        }

        let pos_embedding = if position_encoding == PositionEncoding::Learned {
            Some(Embedding::new(max_seq_len, d_model))
        } else {
            None
        };

        Ok(Self {
            token_embedding: Embedding::new(vocab_size, d_model),
            pos_embedding,
            layers,
            norm: NormLayer::LayerNorm(LayerNorm::new(vec![d_model], eps)),
            output: Linear::new(d_model, vocab_size, false),
            d_model,
            vocab_size,
            max_seq_len,
            n_layers: num_layers,
            n_heads: num_heads,
            n_kv_heads,
            position_encoding,
        })
    }

    /// Construct model from pre-loaded and remapped weights (internal names).
    ///
    /// Expects weights already remapped via `WeightMap::remap()`.
    /// Positional embedding is random-initialized for Learned encoding.
    pub fn from_pretrained(config: &crate::config::ModelConfig, weights: HashMap<String, Tensor>) -> Result<Self> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let vocab_size = config.vocab_size;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let n_kv_heads = config.n_kv_heads.unwrap_or(num_heads);
        let position_encoding = config.position_encoding;
        let causal = config.causal;
        let rope_theta = config.rope_theta;

        // Convert to F32 (for Embedding, LayerNorm — must be float)
        let get_tensor = |key: &str| -> Result<Tensor> {
            weights.get(key)
                .ok_or_else(|| LLMForgeError::NotImplemented(
                    format!("Missing weight: {}", key)
                ))
                .and_then(|t| t.to_f32())
        };

        // Preserve quantized dtype (for Linear weights — Q4_0/Q8_0 stay quantized)
        let get_weight = |key: &str| -> Result<Tensor> {
            weights.get(key)
                .ok_or_else(|| LLMForgeError::NotImplemented(
                    format!("Missing weight: {}", key)
                ))
                .and_then(|t| match t.dtype() {
                    DType::Q4_0 | DType::Q8_0 | DType::F32 => Ok(t.clone()),
                    _ => t.to_f32(),
                })
        };

        // Token embedding (needs F32 for lookup)
        let token_embedding = Embedding::from_weights(get_tensor("token_embedding.weight")?);

        // Positional embedding: only for Learned
        let pos_embedding = if position_encoding == PositionEncoding::Learned {
            if let Ok(pos_weight) = get_tensor("pos_embedding.weight") {
                Some(Embedding::from_weights(pos_weight))
            } else {
                Some(Embedding::new(max_seq_len, d_model))
            }
        } else {
            None
        };

        // Transformer layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let q_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.q_proj.weight", i))?,
                None,
            );
            let k_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.k_proj.weight", i))?,
                None,
            );
            let v_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.v_proj.weight", i))?,
                None,
            );
            let out_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.out_proj.weight", i))?,
                None,
            );
            let attention = MultiHeadAttention::from_weights(
                d_model, num_heads, config.n_kv_heads,
                q_proj, k_proj, v_proj, out_proj,
                causal, position_encoding, max_seq_len, rope_theta,
            )?;

            let up_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.feed_forward.up_proj.weight", i))?,
                None,
            );
            let down_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.feed_forward.down_proj.weight", i))?,
                None,
            );

            // Detect SwiGLU: if gate_proj weight is present
            let gate_key = format!("layers.{}.feed_forward.gate_proj.weight", i);
            let feed_forward = if let Ok(gate_weight) = get_weight(&gate_key) {
                let gate_proj = Linear::from_weights(gate_weight, None);
                FeedForward::from_weights_swiglu(up_proj, gate_proj, down_proj)
            } else {
                FeedForward::from_weights_with_activation(up_proj, down_proj, Activation::Silu)
            };

            // RMSNorm for Llama-family models (weight only, no bias, no mean subtraction)
            let attention_norm = RMSNorm::from_weight(
                get_tensor(&format!("layers.{}.attention_norm.weight", i))?,
                eps,
            );
            let ffn_norm = RMSNorm::from_weight(
                get_tensor(&format!("layers.{}.ffn_norm.weight", i))?,
                eps,
            );

            layers.push(TransformerBlock::from_weights_rms(attention, feed_forward, attention_norm, ffn_norm));
        }

        // Final RMSNorm (F32)
        let norm = NormLayer::RMSNorm(RMSNorm::from_weight(get_tensor("norm.weight")?, eps));

        // Output projection (can be quantized)
        let output = Linear::from_weights(get_weight("output.weight")?, None);

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
            n_kv_heads,
            position_encoding,
        })
    }

    /// Construct GPT-2 model from pre-loaded and remapped weights.
    ///
    /// Handles GPT-2 specifics:
    /// - Tied embeddings: output weight = token_embedding weight (clone)
    /// - Fused QKV: c_attn split into separate Q/K/V projections
    /// - Positional embedding loaded from weights
    /// - LayerNorm with bias
    /// - GELU activation (no gate_proj)
    pub fn from_pretrained_gpt2(config: &crate::config::ModelConfig, weights: HashMap<String, Tensor>) -> Result<Self> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let vocab_size = config.vocab_size;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let n_kv_heads = config.n_kv_heads.unwrap_or(num_heads);
        let position_encoding = config.position_encoding;
        let causal = config.causal;
        let rope_theta = config.rope_theta;

        let get_tensor = |key: &str| -> Result<Tensor> {
            weights.get(key)
                .ok_or_else(|| LLMForgeError::NotImplemented(
                    format!("Missing weight: {}", key)
                ))
                .and_then(|t| t.to_f32())
        };

        let get_tensor_opt = |key: &str| -> Option<Tensor> {
            weights.get(key).and_then(|t| t.to_f32().ok())
        };

        // Token embedding
        let token_embedding = Embedding::from_weights(get_tensor("token_embedding.weight")?);

        // Positional embedding from weights
        let pos_embedding = if position_encoding == PositionEncoding::Learned {
            if let Ok(pos_weight) = get_tensor("pos_embedding.weight") {
                Some(Embedding::from_weights(pos_weight))
            } else {
                Some(Embedding::new(max_seq_len, d_model))
            }
        } else {
            None
        };

        // Transformer layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // GPT-2 uses Conv1D which stores weights as [In, Out].
            // Our Linear expects [Out, In], so transpose all weight matrices.
            let c_attn_weight = get_tensor(&format!("layers.{}.attention.c_attn.weight", i))?
                .transpose(0, 1)?
                .contiguous()?;
            let c_attn_bias = get_tensor_opt(&format!("layers.{}.attention.c_attn.bias", i));

            // Split fused QKV: [3*d_model, d_model] → 3x [d_model, d_model]
            let (q_w, k_w, v_w) = split_qkv(c_attn_weight, d_model)?;
            let (q_b, k_b, v_b) = if let Some(bias) = c_attn_bias {
                let (qb, kb, vb) = split_qkv_bias(bias, d_model)?;
                (Some(qb), Some(kb), Some(vb))
            } else {
                (None, None, None)
            };

            let q_proj = Linear::from_weights(q_w, q_b);
            let k_proj = Linear::from_weights(k_w, k_b);
            let v_proj = Linear::from_weights(v_w, v_b);

            let out_proj_weight = get_tensor(&format!("layers.{}.attention.out_proj.weight", i))?
                .transpose(0, 1)?
                .contiguous()?;
            let out_proj = Linear::from_weights(
                out_proj_weight,
                get_tensor_opt(&format!("layers.{}.attention.out_proj.bias", i)),
            );

            let attention = MultiHeadAttention::from_weights(
                d_model, num_heads, config.n_kv_heads,
                q_proj, k_proj, v_proj, out_proj,
                causal, position_encoding, max_seq_len, rope_theta,
            )?;

            // FFN (GELU, no gate_proj) — also transpose Conv1D weights
            let up_proj_weight = get_tensor(&format!("layers.{}.feed_forward.up_proj.weight", i))?
                .transpose(0, 1)?
                .contiguous()?;
            let up_proj = Linear::from_weights(
                up_proj_weight,
                get_tensor_opt(&format!("layers.{}.feed_forward.up_proj.bias", i)),
            );
            let down_proj_weight = get_tensor(&format!("layers.{}.feed_forward.down_proj.weight", i))?
                .transpose(0, 1)?
                .contiguous()?;
            let down_proj = Linear::from_weights(
                down_proj_weight,
                get_tensor_opt(&format!("layers.{}.feed_forward.down_proj.bias", i)),
            );
            let feed_forward = FeedForward::from_weights_with_activation(up_proj, down_proj, Activation::Gelu);

            // LayerNorm with bias
            let attention_norm = LayerNorm::from_weights(
                get_tensor(&format!("layers.{}.attention_norm.weight", i))?,
                get_tensor(&format!("layers.{}.attention_norm.bias", i))?,
                eps,
            );
            let ffn_norm = LayerNorm::from_weights(
                get_tensor(&format!("layers.{}.ffn_norm.weight", i))?,
                get_tensor(&format!("layers.{}.ffn_norm.bias", i))?,
                eps,
            );

            layers.push(TransformerBlock::from_weights(attention, feed_forward, attention_norm, ffn_norm));
        }

        // Final norm with bias (GPT-2 uses standard LayerNorm)
        let norm = NormLayer::LayerNorm(LayerNorm::from_weights(
            get_tensor("norm.weight")?,
            get_tensor("norm.bias")?,
            eps,
        ));

        // Tied embeddings: output weight = token_embedding weight
        let output_weight = token_embedding.weight.clone();
        let output = Linear::from_weights(output_weight, None);

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
            n_kv_heads,
            position_encoding,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // [Batch, Seq] -> [Batch, Seq, D_Model]
        let x_emb = self.token_embedding.forward(input_ids)?;

        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        if seq_len > self.max_seq_len {
             return Err(crate::error::LLMForgeError::IndexOutOfBounds{ index: seq_len, dim: 1, size: self.max_seq_len });
        }

        // Add positional embeddings (only for Learned)
        let mut x = if let Some(ref pos_emb) = self.pos_embedding {
            let mut pos_data = Vec::with_capacity(batch_size * seq_len);
            for _ in 0..batch_size {
                for i in 0..seq_len {
                    pos_data.push(i as f32);
                }
            }
            let pos_bytes = crate::core::tensor::f32_vec_to_bytes(pos_data);
            let pos_ids = Tensor::new(pos_bytes, vec![batch_size, seq_len], DType::F32);
            let p_emb = pos_emb.forward(&pos_ids)?;
            x_emb.add(&p_emb)?
        } else {
            x_emb
        };

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

        // Add positional embeddings (only for Learned)
        let mut x = if let Some(ref pos_emb) = self.pos_embedding {
            let mut pos_data = Vec::with_capacity(batch_size * seq_len);
            for _ in 0..batch_size {
                for i in 0..seq_len {
                    pos_data.push((start_pos + i) as f32);
                }
            }
            let pos_bytes = crate::core::tensor::f32_vec_to_bytes(pos_data);
            let pos_ids = Tensor::new(pos_bytes, vec![batch_size, seq_len], DType::F32);
            let p_emb = pos_emb.forward(&pos_ids)?;
            x_emb.add(&p_emb)?
        } else {
            x_emb
        };

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_with_cache(&x, None, cache, i)?;
        }

        let x = self.norm.forward(&x)?;
        self.output.forward(&x)
    }

    /// Toggle native Q4_0×Q8_0 integer matmul on all Linear layers in the model.
    pub fn set_native_q4_matmul(&mut self, enabled: bool) {
        for layer in &mut self.layers {
            layer.set_native_q4_matmul(enabled);
        }
        self.output.use_native_q4 = enabled;
    }

    /// Freeze token (and optional positional) embeddings.
    pub fn freeze_embeddings(&mut self) {
        self.token_embedding.freeze();
        if let Some(ref mut pos_emb) = self.pos_embedding {
            pos_emb.freeze();
        }
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let mut total = 0usize;
        let mut frozen = 0usize;

        let tok_count = self.token_embedding.weight.element_count();
        total += tok_count;
        if self.token_embedding.frozen { frozen += tok_count; }

        if let Some(ref pos_emb) = self.pos_embedding {
            let count = pos_emb.weight.element_count();
            total += count;
            if pos_emb.frozen { frozen += count; }
        }

        // Transformer layers
        for layer in &self.layers {
            let (t, f) = layer.parameter_count();
            total += t;
            frozen += f;
        }

        // Output projection
        let (t, f) = self.output.parameter_count();
        total += t;
        frozen += f;

        // Final norm
        let (t, f) = self.norm.parameter_count();
        total += t;
        frozen += f;

        (total, frozen)
    }
}

/// Split a fused QKV weight tensor [3*d_model, d_model] into three [d_model, d_model] tensors.
fn split_qkv(qkv: Tensor, d_model: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let q = qkv.slice_rows(0, d_model)?;
    let k = qkv.slice_rows(d_model, 2 * d_model)?;
    let v = qkv.slice_rows(2 * d_model, 3 * d_model)?;
    Ok((q, k, v))
}

/// Split a fused QKV bias tensor [3*d_model] into three [d_model] tensors.
fn split_qkv_bias(qkv_bias: Tensor, d_model: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let data = qkv_bias.as_slice_f32()?;
    if data.len() != 3 * d_model {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![3 * d_model],
            actual: vec![data.len()],
        });
    }

    let q_data = crate::core::tensor::f32_vec_to_bytes(data[..d_model].to_vec());
    let k_data = crate::core::tensor::f32_vec_to_bytes(data[d_model..2 * d_model].to_vec());
    let v_data = crate::core::tensor::f32_vec_to_bytes(data[2 * d_model..].to_vec());

    Ok((
        Tensor::new(q_data, vec![d_model], DType::F32),
        Tensor::new(k_data, vec![d_model], DType::F32),
        Tensor::new(v_data, vec![d_model], DType::F32),
    ))
}
