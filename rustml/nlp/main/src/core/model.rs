//! Unified LLM model supporting GPT-2, Llama, and future architectures.
//!
//! Uses the rustml-nn transformer infrastructure (MultiHeadAttention, TransformerBlock,
//! FeedForward, KVCache, RoPE, etc.) to build a single model struct that can load
//! weights from either SafeTensors (GPT-2) or GGUF (Llama) format.

use crate::api::error::{NlpError, NlpResult};
use crate::api::types::{LanguageModel, ModelConfig};
use crate::core::weight_map::WeightMap;
use std::time::Instant;
use rustml_core::{DType, Tensor, f32_vec_to_bytes};
use rustml_nn::{
    Activation, Embedding, FeedForward, KVCache, LayerNorm, Linear, MoeLayer,
    MultiHeadAttention, NormLayer, PositionEncoding, RMSNorm, RoPEFreqs, TransformerBlock,
};
use std::collections::HashMap;

/// Unified language model supporting multiple architectures.
pub struct LlmModel {
    pub token_embedding: Embedding,
    pub pos_embedding: Option<Embedding>,
    pub layers: Vec<TransformerBlock>,
    pub norm: NormLayer,
    pub output: Linear,
    pub config: ModelConfig,
}

impl LlmModel {
    /// Create a randomly-initialized model from config.
    pub fn new(config: &ModelConfig) -> NlpResult<Self> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let hidden_dim = config.hidden_dim;
        let num_layers = config.n_layers;
        let vocab_size = config.vocab_size;
        let bias = config.use_bias.unwrap_or(false);
        let eps = config.norm_eps;
        let max_seq_len = config.max_seq_len;
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
            norm: NormLayer::LayerNorm(LayerNorm::with_eps(d_model, eps)),
            output: Linear::new_no_bias(d_model, vocab_size),
            config: config.clone(),
        })
    }

    /// Construct from pre-loaded GGUF/Llama weights (already remapped to internal names).
    ///
    /// Expects weight names like:
    /// - `token_embedding.weight`
    /// - `layers.{i}.attention.{q,k,v,out}_proj.weight`
    /// - `layers.{i}.feed_forward.{up,down,gate}_proj.weight`
    /// - `layers.{i}.{attention,ffn}_norm.weight`
    /// - `norm.weight`, `output.weight`
    pub fn from_pretrained(
        config: &ModelConfig,
        weights: HashMap<String, Tensor>,
    ) -> NlpResult<Self> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let position_encoding = config.position_encoding;
        let causal = config.causal;
        let rope_theta = config.rope_theta;

        // F32 conversion for embeddings/norms
        let get_tensor = |key: &str| -> NlpResult<Tensor> {
            weights
                .get(key)
                .ok_or_else(|| NlpError::ModelError(format!("Missing weight: {}", key)))
                .and_then(|t| Ok(t.to_f32()?))
        };

        // Preserve quantized dtype for Linear weights
        let get_weight = |key: &str| -> NlpResult<Tensor> {
            weights
                .get(key)
                .ok_or_else(|| NlpError::ModelError(format!("Missing weight: {}", key)))
                .and_then(|t| match t.dtype() {
                    DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::F32 => Ok(t.clone()),
                    _ => Ok(t.to_f32()?),
                })
        };

        let token_embedding = Embedding::from_weights(get_tensor("token_embedding.weight")?)?;

        let pos_embedding = if position_encoding == PositionEncoding::Learned {
            if let Ok(pos_weight) = get_tensor("pos_embedding.weight") {
                Some(Embedding::from_weights(pos_weight)?)
            } else {
                Some(Embedding::new(max_seq_len, d_model))
            }
        } else {
            None
        };

        let has_attn_bias = config.attention_bias == Some(true);
        let get_bias_opt = |key: &str| -> Option<Tensor> {
            weights.get(key).and_then(|t| t.to_f32().ok())
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let q_bias = if has_attn_bias {
                get_bias_opt(&format!("layers.{}.attention.q_proj.bias", i))
            } else { None };
            let k_bias = if has_attn_bias {
                get_bias_opt(&format!("layers.{}.attention.k_proj.bias", i))
            } else { None };
            let v_bias = if has_attn_bias {
                get_bias_opt(&format!("layers.{}.attention.v_proj.bias", i))
            } else { None };
            let o_bias = if has_attn_bias {
                get_bias_opt(&format!("layers.{}.attention.out_proj.bias", i))
            } else { None };

            let q_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.q_proj.weight", i))?,
                q_bias,
            )?;
            let k_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.k_proj.weight", i))?,
                k_bias,
            )?;
            let v_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.v_proj.weight", i))?,
                v_bias,
            )?;
            let out_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.out_proj.weight", i))?,
                o_bias,
            )?;
            let mut attention = MultiHeadAttention::from_weights(
                d_model,
                num_heads,
                config.n_kv_heads,
                q_proj,
                k_proj,
                v_proj,
                out_proj,
                causal,
                position_encoding,
                max_seq_len,
                rope_theta,
            )?;

            // Sliding window (Mistral: all layers; Gemma-2: even layers only)
            if let Some(window) = config.sliding_window {
                let apply = if config.attn_logit_cap.is_some() { i % 2 == 0 } else { true };
                if apply { attention.set_window_size(window); }
            }
            // Logit capping (Gemma-2)
            if let Some(cap) = config.attn_logit_cap { attention.set_attn_logit_cap(cap); }

            let up_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.feed_forward.up_proj.weight", i))?,
                None,
            )?;
            let down_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.feed_forward.down_proj.weight", i))?,
                None,
            )?;

            // Detect SwiGLU: if gate_proj is present
            let gate_key = format!("layers.{}.feed_forward.gate_proj.weight", i);
            let feed_forward = if let Ok(gate_weight) = get_weight(&gate_key) {
                let gate_proj = Linear::from_weights(gate_weight, None)?;
                FeedForward::from_weights_swiglu(up_proj, gate_proj, down_proj)
            } else {
                FeedForward::from_weights_with_activation(up_proj, down_proj, Activation::Silu)
            };

            // RMSNorm with optional offset (Gemma-2: offset=1.0)
            let offset = config.rms_norm_offset.unwrap_or(0.0);
            let attention_norm = RMSNorm::from_weight_with_offset(
                get_tensor(&format!("layers.{}.attention_norm.weight", i))?,
                eps,
                offset,
            );
            let ffn_norm = RMSNorm::from_weight_with_offset(
                get_tensor(&format!("layers.{}.ffn_norm.weight", i))?,
                eps,
                offset,
            );

            layers.push(TransformerBlock::from_weights_rms(
                attention,
                feed_forward,
                attention_norm,
                ffn_norm,
            ));
        }

        let norm = NormLayer::RMSNorm(RMSNorm::from_weight(get_tensor("norm.weight")?, eps));
        let output = Linear::from_weights(get_weight("output.weight")?, None)?;

        Ok(Self {
            token_embedding,
            pos_embedding,
            layers,
            norm,
            output,
            config: config.clone(),
        })
    }

    /// Construct GPT-2 model from weights (already remapped to internal names).
    ///
    /// Handles GPT-2 specifics:
    /// - Tied embeddings (output weight = token_embedding weight)
    /// - Fused QKV split (c_attn → separate Q/K/V)
    /// - Conv1D transpose (HF stores [in, out], we need [out, in])
    /// - LayerNorm with bias
    /// - GELU activation
    pub fn from_pretrained_gpt2(
        config: &ModelConfig,
        weights: HashMap<String, Tensor>,
    ) -> NlpResult<Self> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let position_encoding = config.position_encoding;
        let causal = config.causal;
        let rope_theta = config.rope_theta;

        let get_tensor = |key: &str| -> NlpResult<Tensor> {
            weights
                .get(key)
                .ok_or_else(|| NlpError::ModelError(format!("Missing weight: {}", key)))
                .and_then(|t| Ok(t.to_f32()?))
        };

        let get_tensor_opt = |key: &str| -> Option<Tensor> {
            weights.get(key).and_then(|t| t.to_f32().ok())
        };

        let token_embedding = Embedding::from_weights(get_tensor("token_embedding.weight")?)?;

        let pos_embedding = if position_encoding == PositionEncoding::Learned {
            if let Ok(pos_weight) = get_tensor("pos_embedding.weight") {
                Some(Embedding::from_weights(pos_weight)?)
            } else {
                Some(Embedding::new(max_seq_len, d_model))
            }
        } else {
            None
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // GPT-2 Conv1D: [In, Out] → transpose to [Out, In]
            let c_attn_weight = get_tensor(&format!("layers.{}.attention.c_attn.weight", i))?
                .transpose(0, 1)?
                .contiguous()?;
            let c_attn_bias =
                get_tensor_opt(&format!("layers.{}.attention.c_attn.bias", i));

            // Split fused QKV
            let (q_w, k_w, v_w) = split_qkv(&c_attn_weight, d_model)?;
            let (q_b, k_b, v_b) = if let Some(bias) = c_attn_bias {
                let (qb, kb, vb) = split_qkv_bias(&bias, d_model)?;
                (Some(qb), Some(kb), Some(vb))
            } else {
                (None, None, None)
            };

            let q_proj = Linear::from_weights(q_w, q_b)?;
            let k_proj = Linear::from_weights(k_w, k_b)?;
            let v_proj = Linear::from_weights(v_w, v_b)?;

            let out_proj_weight =
                get_tensor(&format!("layers.{}.attention.out_proj.weight", i))?
                    .transpose(0, 1)?
                    .contiguous()?;
            let out_proj = Linear::from_weights(
                out_proj_weight,
                get_tensor_opt(&format!("layers.{}.attention.out_proj.bias", i)),
            )?;

            let attention = MultiHeadAttention::from_weights(
                d_model,
                num_heads,
                config.n_kv_heads,
                q_proj,
                k_proj,
                v_proj,
                out_proj,
                causal,
                position_encoding,
                max_seq_len,
                rope_theta,
            )?;

            // FFN (GELU, no gate_proj) — also transpose Conv1D weights
            let up_proj_weight =
                get_tensor(&format!("layers.{}.feed_forward.up_proj.weight", i))?
                    .transpose(0, 1)?
                    .contiguous()?;
            let up_proj = Linear::from_weights(
                up_proj_weight,
                get_tensor_opt(&format!("layers.{}.feed_forward.up_proj.bias", i)),
            )?;
            let down_proj_weight =
                get_tensor(&format!("layers.{}.feed_forward.down_proj.weight", i))?
                    .transpose(0, 1)?
                    .contiguous()?;
            let down_proj = Linear::from_weights(
                down_proj_weight,
                get_tensor_opt(&format!("layers.{}.feed_forward.down_proj.bias", i)),
            )?;
            let feed_forward =
                FeedForward::from_weights_with_activation(up_proj, down_proj, Activation::Gelu);

            // LayerNorm with bias
            let attention_norm = LayerNorm::from_weights(
                get_tensor(&format!("layers.{}.attention_norm.weight", i))?,
                get_tensor(&format!("layers.{}.attention_norm.bias", i))?,
                eps,
            )?;
            let ffn_norm = LayerNorm::from_weights(
                get_tensor(&format!("layers.{}.ffn_norm.weight", i))?,
                get_tensor(&format!("layers.{}.ffn_norm.bias", i))?,
                eps,
            )?;

            layers.push(TransformerBlock::from_weights(
                attention,
                feed_forward,
                attention_norm,
                ffn_norm,
            ));
        }

        // Final LayerNorm with bias
        let norm = NormLayer::LayerNorm(LayerNorm::from_weights(
            get_tensor("norm.weight")?,
            get_tensor("norm.bias")?,
            eps,
        )?);

        // Tied embeddings: output weight = token_embedding weight
        let output = Linear::from_weights(token_embedding.weight.clone(), None)?;

        Ok(Self {
            token_embedding,
            pos_embedding,
            layers,
            norm,
            output,
            config: config.clone(),
        })
    }

    /// Construct Falcon model from weights (already remapped to internal names).
    ///
    /// Handles Falcon specifics:
    /// - Fused QKV split with separate Q/KV dimensions for MQA/GQA
    /// - LayerNorm with bias (not RMSNorm)
    /// - GELU activation (no gate_proj)
    /// - Parallel residual connections
    /// - ALiBi or RoPE from config
    /// - Tied embeddings fallback
    pub fn from_pretrained_falcon(
        config: &ModelConfig,
        weights: HashMap<String, Tensor>,
    ) -> NlpResult<Self> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_kv_heads = config.n_kv_heads.unwrap_or(num_heads);
        let num_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let position_encoding = config.position_encoding;
        let causal = config.causal;
        let rope_theta = config.rope_theta;
        let head_dim = d_model / num_heads;

        let get_tensor = |key: &str| -> NlpResult<Tensor> {
            weights
                .get(key)
                .ok_or_else(|| NlpError::ModelError(format!("Missing weight: {}", key)))
                .and_then(|t| Ok(t.to_f32()?))
        };

        let get_tensor_opt = |key: &str| -> Option<Tensor> {
            weights.get(key).and_then(|t| t.to_f32().ok())
        };

        let token_embedding = Embedding::from_weights(get_tensor("token_embedding.weight")?)?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // Fused QKV: split [q_dim + kv_dim + kv_dim, d_model]
            let qkv_weight = get_tensor(&format!("layers.{}.attention.qkv.weight", i))?;
            let qkv_bias = get_tensor_opt(&format!("layers.{}.attention.qkv.bias", i));

            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let q_w = qkv_weight.slice(0, 0, q_dim)?;
            let k_w = qkv_weight.slice(0, q_dim, q_dim + kv_dim)?;
            let v_w = qkv_weight.slice(0, q_dim + kv_dim, q_dim + 2 * kv_dim)?;

            let (q_b, k_b, v_b) = if let Some(ref bias) = qkv_bias {
                let bias_data: Vec<f32> = bias.iter().collect();
                let q_b_data = f32_vec_to_bytes(bias_data[..q_dim].to_vec());
                let k_b_data = f32_vec_to_bytes(bias_data[q_dim..q_dim + kv_dim].to_vec());
                let v_b_data = f32_vec_to_bytes(bias_data[q_dim + kv_dim..].to_vec());
                (
                    Some(Tensor::new(q_b_data, vec![q_dim], DType::F32)),
                    Some(Tensor::new(k_b_data, vec![kv_dim], DType::F32)),
                    Some(Tensor::new(v_b_data, vec![kv_dim], DType::F32)),
                )
            } else {
                (None, None, None)
            };

            let q_proj = Linear::from_weights(q_w, q_b)?;
            let k_proj = Linear::from_weights(k_w, k_b)?;
            let v_proj = Linear::from_weights(v_w, v_b)?;

            let out_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.attention.out_proj.weight", i))?,
                get_tensor_opt(&format!("layers.{}.attention.out_proj.bias", i)),
            )?;

            let attention = MultiHeadAttention::from_weights(
                d_model,
                num_heads,
                config.n_kv_heads,
                q_proj,
                k_proj,
                v_proj,
                out_proj,
                causal,
                position_encoding,
                max_seq_len,
                rope_theta,
            )?;

            // GELU FFN (no gate_proj)
            let up_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.feed_forward.up_proj.weight", i))?,
                get_tensor_opt(&format!("layers.{}.feed_forward.up_proj.bias", i)),
            )?;
            let down_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.feed_forward.down_proj.weight", i))?,
                get_tensor_opt(&format!("layers.{}.feed_forward.down_proj.bias", i)),
            )?;
            let feed_forward =
                FeedForward::from_weights_with_activation(up_proj, down_proj, Activation::Gelu);

            // LayerNorm with bias
            let attention_norm = LayerNorm::from_weights(
                get_tensor(&format!("layers.{}.attention_norm.weight", i))?,
                get_tensor(&format!("layers.{}.attention_norm.bias", i))?,
                eps,
            )?;
            let ffn_norm = LayerNorm::from_weights(
                get_tensor(&format!("layers.{}.ffn_norm.weight", i))?,
                get_tensor(&format!("layers.{}.ffn_norm.bias", i))?,
                eps,
            )?;

            let mut block = TransformerBlock::from_weights(
                attention,
                feed_forward,
                attention_norm,
                ffn_norm,
            );
            block.set_parallel_residual(config.parallel_residual.unwrap_or(true));

            layers.push(block);
        }

        // Final LayerNorm with bias
        let norm = NormLayer::LayerNorm(LayerNorm::from_weights(
            get_tensor("norm.weight")?,
            get_tensor("norm.bias")?,
            eps,
        )?);

        // Tied embeddings fallback if output.weight missing
        let output = if let Ok(w) = get_tensor("output.weight") {
            Linear::from_weights(w, None)?
        } else {
            Linear::from_weights(token_embedding.weight.clone(), None)?
        };

        Ok(Self {
            token_embedding,
            pos_embedding: None,
            layers,
            norm,
            output,
            config: config.clone(),
        })
    }

    /// Construct Mixtral MoE model from weights (already remapped to internal names).
    ///
    /// Handles Mixtral specifics:
    /// - Attention follows Llama + sliding window
    /// - Per-layer: router gate + N expert SwiGLU FFNs → MoeLayer
    /// - TransformerBlock's feed_forward is unused (MoE path in forward takes precedence)
    pub fn from_pretrained_mixtral(
        config: &ModelConfig,
        weights: HashMap<String, Tensor>,
    ) -> NlpResult<Self> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let position_encoding = config.position_encoding;
        let causal = config.causal;
        let rope_theta = config.rope_theta;
        let num_experts = config.num_local_experts.unwrap_or(8);
        let num_experts_per_tok = config.num_experts_per_tok.unwrap_or(2);

        let get_tensor = |key: &str| -> NlpResult<Tensor> {
            weights
                .get(key)
                .ok_or_else(|| NlpError::ModelError(format!("Missing weight: {}", key)))
                .and_then(|t| Ok(t.to_f32()?))
        };

        let get_weight = |key: &str| -> NlpResult<Tensor> {
            weights
                .get(key)
                .ok_or_else(|| NlpError::ModelError(format!("Missing weight: {}", key)))
                .and_then(|t| match t.dtype() {
                    DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::F32 => Ok(t.clone()),
                    _ => Ok(t.to_f32()?),
                })
        };

        let token_embedding = Embedding::from_weights(get_tensor("token_embedding.weight")?)?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // Attention: same as Llama
            let q_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.q_proj.weight", i))?,
                None,
            )?;
            let k_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.k_proj.weight", i))?,
                None,
            )?;
            let v_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.v_proj.weight", i))?,
                None,
            )?;
            let out_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.out_proj.weight", i))?,
                None,
            )?;
            let mut attention = MultiHeadAttention::from_weights(
                d_model,
                num_heads,
                config.n_kv_heads,
                q_proj,
                k_proj,
                v_proj,
                out_proj,
                causal,
                position_encoding,
                max_seq_len,
                rope_theta,
            )?;

            // Sliding window
            if let Some(window) = config.sliding_window {
                attention.set_window_size(window);
            }

            // MoE: router gate + expert FFNs
            let gate = Linear::from_weights(
                get_weight(&format!("layers.{}.moe.gate.weight", i))?,
                None,
            )?;

            let mut experts = Vec::with_capacity(num_experts);
            for j in 0..num_experts {
                let gate_proj = Linear::from_weights(
                    get_weight(&format!("layers.{}.moe.experts.{}.gate_proj.weight", i, j))?,
                    None,
                )?;
                let up_proj = Linear::from_weights(
                    get_weight(&format!("layers.{}.moe.experts.{}.up_proj.weight", i, j))?,
                    None,
                )?;
                let down_proj = Linear::from_weights(
                    get_weight(&format!("layers.{}.moe.experts.{}.down_proj.weight", i, j))?,
                    None,
                )?;
                experts.push(FeedForward::from_weights_swiglu(up_proj, gate_proj, down_proj));
            }

            let moe_layer = MoeLayer::from_weights(gate, experts, num_experts_per_tok);

            // RMSNorm
            let attention_norm = RMSNorm::from_weight(
                get_tensor(&format!("layers.{}.attention_norm.weight", i))?,
                eps,
            );
            let ffn_norm = RMSNorm::from_weight(
                get_tensor(&format!("layers.{}.ffn_norm.weight", i))?,
                eps,
            );

            // Placeholder feed_forward (unused — MoE path takes precedence)
            let placeholder_ff = FeedForward::swiglu(d_model, config.hidden_dim, false);

            let mut block = TransformerBlock::from_weights_rms(
                attention,
                placeholder_ff,
                attention_norm,
                ffn_norm,
            );
            block.moe = Some(moe_layer);

            layers.push(block);
        }

        let norm = NormLayer::RMSNorm(RMSNorm::from_weight(get_tensor("norm.weight")?, eps));
        let output = Linear::from_weights(get_weight("output.weight")?, None)?;

        Ok(Self {
            token_embedding,
            pos_embedding: None,
            layers,
            norm,
            output,
            config: config.clone(),
        })
    }

    /// Construct Gemma 3 model from weights (already remapped to internal names).
    ///
    /// Handles Gemma 3 specifics:
    /// - Decoupled head_dim (independent of dim/n_heads)
    /// - Per-layer RoPE: local layers use rope_local_base_freq, global layers use rope_theta
    /// - Sliding window pattern: every Nth layer is global attention, rest are local
    /// - Custom attention scaling via query_pre_attn_scalar
    /// - GeGLU activation (gelu(gate) * up)
    /// - RMSNorm with offset=1.0, embedding scale=sqrt(dim)
    /// - Linear RoPE frequency scaling for long context
    pub fn from_pretrained_gemma3(
        config: &ModelConfig,
        weights: HashMap<String, Tensor>,
    ) -> NlpResult<Self> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let causal = config.causal;

        let head_dim = config.head_dim.unwrap_or(d_model / num_heads);
        let attn_scale = config.query_pre_attn_scalar.map(|s| s.sqrt());
        let pattern = config.sliding_window_pattern.unwrap_or(6);
        let global_theta = config.rope_theta;
        let local_theta = config.rope_local_base_freq.unwrap_or(10000.0);
        let scaling = config.rope_scaling_factor.unwrap_or(1.0);

        // Pre-build two RoPEFreqs: one for local layers, one for global layers
        let local_rope = RoPEFreqs::with_scaling(head_dim, max_seq_len, local_theta, scaling);
        let global_rope = RoPEFreqs::with_scaling(head_dim, max_seq_len, global_theta, scaling);

        let get_tensor = |key: &str| -> NlpResult<Tensor> {
            weights
                .get(key)
                .ok_or_else(|| NlpError::ModelError(format!("Missing weight: {}", key)))
                .and_then(|t| Ok(t.to_f32()?))
        };

        let get_weight = |key: &str| -> NlpResult<Tensor> {
            weights
                .get(key)
                .ok_or_else(|| NlpError::ModelError(format!("Missing weight: {}", key)))
                .and_then(|t| match t.dtype() {
                    DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::F32 => Ok(t.clone()),
                    _ => Ok(t.to_f32()?),
                })
        };

        let token_embedding = Embedding::from_weights(get_tensor("token_embedding.weight")?)?;

        let offset = config.rms_norm_offset.unwrap_or(1.0);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let is_global = (i + 1) % pattern == 0;
            let rope = if is_global { global_rope.clone() } else { local_rope.clone() };

            let q_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.q_proj.weight", i))?,
                None,
            )?;
            let k_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.k_proj.weight", i))?,
                None,
            )?;
            let v_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.v_proj.weight", i))?,
                None,
            )?;
            let out_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.attention.out_proj.weight", i))?,
                None,
            )?;

            let mut attention = MultiHeadAttention::from_weights_with_head_dim(
                d_model,
                num_heads,
                config.n_kv_heads,
                head_dim,
                q_proj,
                k_proj,
                v_proj,
                out_proj,
                causal,
                Some(rope),
                attn_scale,
            )?;

            // QK normalization (Gemma 3)
            let q_norm_key = format!("layers.{}.attention.q_norm.weight", i);
            let k_norm_key = format!("layers.{}.attention.k_norm.weight", i);
            if let (Ok(qn_w), Ok(kn_w)) = (get_tensor(&q_norm_key), get_tensor(&k_norm_key)) {
                attention.set_qk_norms(
                    RMSNorm::from_weight_with_offset(qn_w, eps, offset),
                    RMSNorm::from_weight_with_offset(kn_w, eps, offset),
                );
            }

            // Sliding window on local layers only
            if !is_global {
                if let Some(w) = config.sliding_window {
                    attention.set_window_size(w);
                }
            }

            // GeGLU FFN
            let up_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.feed_forward.up_proj.weight", i))?,
                None,
            )?;
            let gate_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.feed_forward.gate_proj.weight", i))?,
                None,
            )?;
            let down_proj = Linear::from_weights(
                get_weight(&format!("layers.{}.feed_forward.down_proj.weight", i))?,
                None,
            )?;
            let feed_forward = FeedForward::from_weights_geglu(up_proj, gate_proj, down_proj);

            // 4 RMSNorms with offset (Gemma 3 sandwich norm)
            let attention_norm = RMSNorm::from_weight_with_offset(
                get_tensor(&format!("layers.{}.attention_norm.weight", i))?,
                eps,
                offset,
            );
            let ffn_norm = RMSNorm::from_weight_with_offset(
                get_tensor(&format!("layers.{}.ffn_norm.weight", i))?,
                eps,
                offset,
            );
            let post_attention_norm = RMSNorm::from_weight_with_offset(
                get_tensor(&format!("layers.{}.post_attention_norm.weight", i))?,
                eps,
                offset,
            );
            let post_ffn_norm = RMSNorm::from_weight_with_offset(
                get_tensor(&format!("layers.{}.post_ffn_norm.weight", i))?,
                eps,
                offset,
            );

            layers.push(TransformerBlock::from_weights_rms_4norm(
                attention,
                feed_forward,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
            ));
        }

        let norm = NormLayer::RMSNorm(RMSNorm::from_weight_with_offset(
            get_tensor("norm.weight")?,
            eps,
            offset,
        ));

        // Gemma 3 uses tied embeddings: output weight = token_embedding weight
        let output = if let Ok(w) = get_weight("output.weight") {
            Linear::from_weights(w, None)?
        } else {
            Linear::from_weights(token_embedding.weight.clone(), None)?
        };

        Ok(Self {
            token_embedding,
            pos_embedding: None,
            layers,
            norm,
            output,
            config: config.clone(),
        })
    }

    /// Quantize the lm_head (output projection) weight from F32 to Q8_0.
    /// Reduces memory bandwidth ~4x for large-vocabulary models.
    /// No-op if weight is already quantized or alignment requirements aren't met.
    pub fn quantize_lm_head(&mut self) -> NlpResult<()> {
        self.output.quantize_weight_q8()?;
        Ok(())
    }

    /// Quantize all F32 linear layers to Q8_0 at load time.
    /// Covers attention projections, FFN projections, MoE experts, and lm_head.
    /// Returns the number of layers successfully quantized.
    /// Safe no-op for non-F32 or already-quantized weights.
    pub fn quantize_all_weights(&mut self) -> NlpResult<usize> {
        let mut count = 0usize;
        fn try_q(l: &mut Linear, c: &mut usize) {
            let was = l.is_quantized();
            if l.quantize_weight_q8().is_ok() && !was && l.is_quantized() {
                *c += 1;
            }
        }
        for layer in &mut self.layers {
            try_q(&mut layer.attention.q_proj, &mut count);
            try_q(&mut layer.attention.k_proj, &mut count);
            try_q(&mut layer.attention.v_proj, &mut count);
            try_q(&mut layer.attention.out_proj, &mut count);
            try_q(&mut layer.feed_forward.up_proj, &mut count);
            try_q(&mut layer.feed_forward.down_proj, &mut count);
            if let Some(ref mut g) = layer.feed_forward.gate_proj {
                try_q(g, &mut count);
            }
            if let Some(ref mut moe) = layer.moe {
                try_q(&mut moe.gate, &mut count);
                for expert in &mut moe.experts {
                    try_q(&mut expert.up_proj, &mut count);
                    try_q(&mut expert.down_proj, &mut count);
                    if let Some(ref mut g) = expert.gate_proj {
                        try_q(g, &mut count);
                    }
                }
            }
        }
        try_q(&mut self.output, &mut count);
        Ok(count)
    }

    /// Apply an optimization profile across all layers.
    ///
    /// Sets `use_inplace_ops` on each TransformerBlock and
    /// `use_inplace_scaling` on each attention layer.
    pub fn set_optimization_profile(&mut self, profile: rustml_core::OptProfile) {
        let inplace = profile.use_inplace_ops();
        for layer in &mut self.layers {
            layer.set_use_inplace_ops(inplace);
            layer.attention.set_use_inplace_scaling(inplace);
        }
    }

    /// Forward pass without KV cache.
    pub fn forward_pass(&self, input_ids: &Tensor) -> NlpResult<Tensor> {
        let x_emb = self.token_embedding.forward(input_ids)?;
        let x_emb = if let Some(scale) = self.config.embedding_scale {
            x_emb.mul_scalar(scale)
        } else {
            x_emb
        };

        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[input_ids.ndim() - 1];

        if seq_len > self.config.max_seq_len {
            return Err(NlpError::ModelError(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.config.max_seq_len
            )));
        }

        // Add positional embeddings (Learned only)
        let mut x = if let Some(ref pos_emb) = self.pos_embedding {
            let mut pos_data = Vec::with_capacity(batch_size * seq_len);
            for _ in 0..batch_size {
                for i in 0..seq_len {
                    pos_data.push(i as f32);
                }
            }
            let pos_bytes = f32_vec_to_bytes(pos_data);
            let pos_ids = Tensor::new(pos_bytes, vec![batch_size, seq_len], DType::F32);
            let p_emb = pos_emb.forward(&pos_ids)?;
            x_emb.add(&p_emb)?
        } else {
            x_emb
        };

        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        let x = self.norm.forward(&x)?;
        Ok(self.output.forward(&x)?)
    }

    /// Forward pass with KV cache for autoregressive decoding.
    pub fn forward_with_cache_pass(
        &self,
        input_ids: &Tensor,
        cache: &mut KVCache,
    ) -> NlpResult<Tensor> {
        let _t_total = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };

        let _t_emb = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
        let x_emb = self.token_embedding.forward(input_ids)?;
        let x_emb = if let Some(scale) = self.config.embedding_scale {
            x_emb.mul_scalar(scale)
        } else {
            x_emb
        };
        let emb_ms = _t_emb.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[input_ids.ndim() - 1];
        let start_pos = cache.current_len;

        if start_pos + seq_len > self.config.max_seq_len {
            return Err(NlpError::ModelError(format!(
                "Sequence length {} exceeds maximum {} (start_pos={})",
                start_pos + seq_len,
                self.config.max_seq_len,
                start_pos
            )));
        }

        // Add positional embeddings with offset (Learned only)
        let mut x = if let Some(ref pos_emb) = self.pos_embedding {
            let mut pos_data = Vec::with_capacity(batch_size * seq_len);
            for _ in 0..batch_size {
                for i in 0..seq_len {
                    pos_data.push((start_pos + i) as f32);
                }
            }
            let pos_bytes = f32_vec_to_bytes(pos_data);
            let pos_ids = Tensor::new(pos_bytes, vec![batch_size, seq_len], DType::F32);
            let p_emb = pos_emb.forward(&pos_ids)?;
            x_emb.add(&p_emb)?
        } else {
            x_emb
        };

        let _t_layers = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
        for (i, layer) in self.layers.iter().enumerate() {
            let _t_layer = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
            x = layer.forward_with_cache(&x, None, cache, i)?;
            if let Some(t) = _t_layer {
                log::debug!("[perf] model::forward layer={} total={:.3}ms", i, t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let layers_ms = _t_layers.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

        let _t_norm = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
        let x = self.norm.forward(&x)?;
        let norm_ms = _t_norm.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

        let _t_proj = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
        let out = self.output.forward(&x)?;
        let proj_ms = _t_proj.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

        if let Some(t) = _t_total {
            log::debug!("[perf] model::forward embedding={:.3}ms layers={:.3}ms norm={:.3}ms projection={:.3}ms total={:.3}ms",
                emb_ms, layers_ms, norm_ms, proj_ms, t.elapsed().as_secs_f64() * 1000.0);
        }

        Ok(out)
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let mut total = 0usize;
        let mut frozen = 0usize;

        total += self.token_embedding.weight.numel();

        if let Some(ref pos_emb) = self.pos_embedding {
            total += pos_emb.weight.numel();
        }

        for layer in &self.layers {
            let (t, f) = layer.parameter_count();
            total += t;
            frozen += f;
        }

        let (t, f) = self.output.parameter_count();
        total += t;
        frozen += f;

        let (t, f) = self.norm.parameter_count();
        total += t;
        frozen += f;

        (total, frozen)
    }
}

impl LanguageModel for LlmModel {
    fn forward(&self, input_ids: &Tensor) -> NlpResult<Tensor> {
        self.forward_pass(input_ids)
    }

    fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KVCache) -> NlpResult<Tensor> {
        self.forward_with_cache_pass(input_ids, cache)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn max_sequence_length(&self) -> usize {
        self.config.max_seq_len
    }

    fn embedding_dim(&self) -> usize {
        self.config.dim
    }

    fn num_layers(&self) -> usize {
        self.config.n_layers
    }

    fn num_kv_heads(&self) -> usize {
        self.config.n_kv_heads.unwrap_or(self.config.n_heads)
    }

    fn head_dim(&self) -> usize {
        self.config.head_dim.unwrap_or(self.config.dim / self.config.n_heads)
    }
}

// ======================== Helper functions ========================

/// Split a fused QKV weight [3*d_model, d_model] into three [d_model, d_model] tensors.
fn split_qkv(qkv: &Tensor, d_model: usize) -> NlpResult<(Tensor, Tensor, Tensor)> {
    let q = qkv.slice(0, 0, d_model)?;
    let k = qkv.slice(0, d_model, 2 * d_model)?;
    let v = qkv.slice(0, 2 * d_model, 3 * d_model)?;
    Ok((q, k, v))
}

/// Split a fused QKV bias [3*d_model] into three [d_model] tensors.
fn split_qkv_bias(qkv_bias: &Tensor, d_model: usize) -> NlpResult<(Tensor, Tensor, Tensor)> {
    let data: Vec<f32> = qkv_bias.iter().collect();
    if data.len() != 3 * d_model {
        return Err(NlpError::ModelError(format!(
            "QKV bias has {} elements, expected {}",
            data.len(),
            3 * d_model
        )));
    }

    let q_data = f32_vec_to_bytes(data[..d_model].to_vec());
    let k_data = f32_vec_to_bytes(data[d_model..2 * d_model].to_vec());
    let v_data = f32_vec_to_bytes(data[2 * d_model..].to_vec());

    Ok((
        Tensor::new(q_data, vec![d_model], DType::F32),
        Tensor::new(k_data, vec![d_model], DType::F32),
        Tensor::new(v_data, vec![d_model], DType::F32),
    ))
}

/// Map HuggingFace GPT-2 weight names to LlmModel internal names.
///
/// | HF Name                                    | Internal Name                                 |
/// |---------------------------------------------|-----------------------------------------------|
/// | transformer.wte.weight                      | token_embedding.weight                        |
/// | transformer.wpe.weight                      | pos_embedding.weight                          |
/// | transformer.h.{i}.ln_1.weight               | layers.{i}.attention_norm.weight               |
/// | transformer.h.{i}.attn.c_attn.weight        | layers.{i}.attention.c_attn.weight              |
/// | transformer.h.{i}.attn.c_proj.weight        | layers.{i}.attention.out_proj.weight            |
/// | transformer.h.{i}.ln_2.weight               | layers.{i}.ffn_norm.weight                     |
/// | transformer.h.{i}.mlp.c_fc.weight           | layers.{i}.feed_forward.up_proj.weight          |
/// | transformer.h.{i}.mlp.c_proj.weight         | layers.{i}.feed_forward.down_proj.weight        |
/// | transformer.ln_f.weight                     | norm.weight                                   |
pub fn map_gpt2_weights(
    weights: HashMap<String, Tensor>,
) -> HashMap<String, Tensor> {
    let mut mapped = HashMap::new();

    for (name, tensor) in weights {
        let stripped = name
            .strip_prefix("transformer.")
            .unwrap_or(&name);

        let new_name = if stripped == "wte.weight" {
            "token_embedding.weight".to_string()
        } else if stripped == "wpe.weight" {
            "pos_embedding.weight".to_string()
        } else if stripped == "ln_f.weight" {
            "norm.weight".to_string()
        } else if stripped == "ln_f.bias" {
            "norm.bias".to_string()
        } else if let Some(rest) = stripped.strip_prefix("h.") {
            // h.{i}.ln_1.* -> layers.{i}.attention_norm.*
            // h.{i}.ln_2.* -> layers.{i}.ffn_norm.*
            // h.{i}.attn.c_attn.* -> layers.{i}.attention.c_attn.*
            // h.{i}.attn.c_proj.* -> layers.{i}.attention.out_proj.*
            // h.{i}.mlp.c_fc.* -> layers.{i}.feed_forward.up_proj.*
            // h.{i}.mlp.c_proj.* -> layers.{i}.feed_forward.down_proj.*
            let parts: Vec<&str> = rest.splitn(2, '.').collect();
            if parts.len() != 2 {
                mapped.insert(name, tensor);
                continue;
            }
            let layer_num = parts[0];
            let suffix = parts[1];

            let mapped_suffix = if let Some(s) = suffix.strip_prefix("ln_1.") {
                format!("attention_norm.{}", s)
            } else if let Some(s) = suffix.strip_prefix("ln_2.") {
                format!("ffn_norm.{}", s)
            } else if let Some(s) = suffix.strip_prefix("attn.c_attn.") {
                format!("attention.c_attn.{}", s)
            } else if let Some(s) = suffix.strip_prefix("attn.c_proj.") {
                format!("attention.out_proj.{}", s)
            } else if let Some(s) = suffix.strip_prefix("mlp.c_fc.") {
                format!("feed_forward.up_proj.{}", s)
            } else if let Some(s) = suffix.strip_prefix("mlp.c_proj.") {
                format!("feed_forward.down_proj.{}", s)
            } else {
                suffix.to_string()
            };

            format!("layers.{}.{}", layer_num, mapped_suffix)
        } else {
            stripped.to_string()
        };

        mapped.insert(new_name, tensor);
    }

    mapped
}

/// Build a SafeTensors model by dispatching on `model_type` from config.json.
///
/// Handles weight remapping and constructor selection for all supported architectures.
pub fn build_safetensors_model(
    model_type: &str,
    config: &ModelConfig,
    weights: HashMap<String, Tensor>,
) -> NlpResult<LlmModel> {
    match model_type {
        "gpt2" | "" => {
            let weights = map_gpt2_weights(weights);
            LlmModel::from_pretrained_gpt2(config, weights)
        }
        "llama" => {
            let wm = WeightMap::llama2(config.n_layers);
            let weights = wm.remap(weights);
            LlmModel::from_pretrained(config, weights)
        }
        "mistral" | "qwen2" | "phi3" => {
            let wm = if config.attention_bias.unwrap_or(false) {
                WeightMap::llama2_with_attn_bias(config.n_layers)
            } else {
                WeightMap::llama2(config.n_layers)
            };
            let weights = wm.remap(weights);
            LlmModel::from_pretrained(config, weights)
        }
        "gemma3" | "gemma3_text" => {
            let wm = WeightMap::gemma3(config.n_layers);
            let weights = wm.remap(weights);
            LlmModel::from_pretrained_gemma3(config, weights)
        }
        "falcon" => {
            let wm = WeightMap::falcon(config.n_layers);
            let weights = wm.remap(weights);
            LlmModel::from_pretrained_falcon(config, weights)
        }
        "mixtral" => {
            let n_experts = config.num_local_experts.unwrap_or(8);
            let wm = WeightMap::mixtral(config.n_layers, n_experts);
            let weights = wm.remap(weights);
            LlmModel::from_pretrained_mixtral(config, weights)
        }
        other => Err(NlpError::ModelError(
            format!("Unsupported SafeTensors model_type: '{}'", other),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            dim: 64,
            hidden_dim: 256,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: None,
            vocab_size: 100,
            norm_eps: 1e-5,
            max_seq_len: 32,
            use_bias: Some(false),
            position_encoding: PositionEncoding::Learned,
            causal: true,
            rope_theta: 10000.0,
            bos_token_id: None,
            eos_token_id: None,
            chat_template: None,
            sliding_window: None,
            attn_logit_cap: None,
            embedding_scale: None,
            rms_norm_offset: None,
            attention_bias: None,
            parallel_residual: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            head_dim: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            rope_local_base_freq: None,
            rope_scaling_factor: None,
        }
    }

    fn tiny_rope_config() -> ModelConfig {
        ModelConfig {
            position_encoding: PositionEncoding::RoPE,
            use_bias: Some(false),
            ..tiny_config()
        }
    }

    #[test]
    fn test_llm_model_creation() {
        let config = tiny_config();
        let model = LlmModel::new(&config).unwrap();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.vocab_size(), 100);
        assert_eq!(model.max_sequence_length(), 32);
    }

    #[test]
    fn test_llm_model_forward() {
        let config = tiny_config();
        let model = LlmModel::new(&config).unwrap();

        let input = Tensor::from_vec(
            (0..8).map(|i| (i % 100) as f32).collect(),
            vec![1, 8],
        )
        .unwrap();

        let logits = model.forward(&input).unwrap();
        assert_eq!(logits.shape(), &[1, 8, 100]);
    }

    #[test]
    fn test_llm_model_rope_forward() {
        let config = tiny_rope_config();
        let model = LlmModel::new(&config).unwrap();

        let input = Tensor::from_vec(
            (0..8).map(|i| (i % 100) as f32).collect(),
            vec![1, 8],
        )
        .unwrap();

        let logits = model.forward(&input).unwrap();
        assert_eq!(logits.shape(), &[1, 8, 100]);
    }

    #[test]
    fn test_llm_model_with_cache() {
        let config = tiny_config();
        let model = LlmModel::new(&config).unwrap();

        let mut cache = KVCache::new(
            config.n_layers,
            config.max_seq_len,
            model.head_dim(),
            model.num_kv_heads(),
        );

        // Prefill with 4 tokens
        let input = Tensor::from_vec(
            (0..4).map(|i| (i % 100) as f32).collect(),
            vec![1, 4],
        )
        .unwrap();
        let logits = model.forward_with_cache(&input, &mut cache).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 100]);
        cache.advance(4);

        // Decode 1 token
        let input = Tensor::from_vec(vec![5.0], vec![1, 1]).unwrap();
        let logits = model.forward_with_cache(&input, &mut cache).unwrap();
        assert_eq!(logits.shape(), &[1, 1, 100]);
    }

    #[test]
    fn test_llm_model_parameter_count() {
        let config = tiny_config();
        let model = LlmModel::new(&config).unwrap();
        let (total, frozen) = model.parameter_count();
        assert!(total > 0);
        assert_eq!(frozen, 0);
    }

    #[test]
    fn test_model_config_validation() {
        let mut config = tiny_config();
        config.dim = 0;
        assert!(config.validate().is_err());

        let mut config = tiny_config();
        config.n_heads = 0;
        assert!(config.validate().is_err());

        let mut config = tiny_config();
        config.dim = 65; // not divisible by n_heads=4
        assert!(config.validate().is_err());

        let mut config = tiny_config();
        config.n_kv_heads = Some(3); // n_heads=4 not divisible by 3
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.dim, 4096);
        assert_eq!(config.n_layers, 32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_split_qkv_fn() {
        let qkv = Tensor::randn(vec![192, 64]); // 3*64, 64
        let (q, k, v) = split_qkv(&qkv, 64).unwrap();
        assert_eq!(q.shape(), &[64, 64]);
        assert_eq!(k.shape(), &[64, 64]);
        assert_eq!(v.shape(), &[64, 64]);
    }

    #[test]
    fn test_split_qkv_bias_fn() {
        let bias = Tensor::from_vec((0..192).map(|i| i as f32).collect(), vec![192]).unwrap();
        let (q, k, v) = split_qkv_bias(&bias, 64).unwrap();
        assert_eq!(q.shape(), &[64]);
        assert_eq!(k.shape(), &[64]);
        assert_eq!(v.shape(), &[64]);
        // Check values
        assert_eq!(q.get(&[0]).unwrap(), 0.0);
        assert_eq!(k.get(&[0]).unwrap(), 64.0);
        assert_eq!(v.get(&[0]).unwrap(), 128.0);
    }

    #[test]
    fn test_map_gpt2_weights_fn() {
        let mut weights = HashMap::new();
        weights.insert(
            "transformer.wte.weight".to_string(),
            Tensor::randn(vec![100, 64]),
        );
        weights.insert(
            "transformer.wpe.weight".to_string(),
            Tensor::randn(vec![32, 64]),
        );
        weights.insert(
            "transformer.h.0.ln_1.weight".to_string(),
            Tensor::randn(vec![64]),
        );
        weights.insert(
            "transformer.h.0.attn.c_attn.weight".to_string(),
            Tensor::randn(vec![64, 192]),
        );
        weights.insert(
            "transformer.h.0.mlp.c_fc.weight".to_string(),
            Tensor::randn(vec![64, 256]),
        );
        weights.insert(
            "transformer.ln_f.weight".to_string(),
            Tensor::randn(vec![64]),
        );

        let mapped = map_gpt2_weights(weights);

        assert!(mapped.contains_key("token_embedding.weight"));
        assert!(mapped.contains_key("pos_embedding.weight"));
        assert!(mapped.contains_key("layers.0.attention_norm.weight"));
        assert!(mapped.contains_key("layers.0.attention.c_attn.weight"));
        assert!(mapped.contains_key("layers.0.feed_forward.up_proj.weight"));
        assert!(mapped.contains_key("norm.weight"));
    }

    #[test]
    fn test_gemma2_embedding_scale() {
        let config = ModelConfig {
            embedding_scale: Some(8.0), // sqrt(64)
            position_encoding: PositionEncoding::RoPE,
            ..tiny_config()
        };
        let model = LlmModel::new(&config).unwrap();

        let input = Tensor::from_vec(
            (0..4).map(|i| (i % 100) as f32).collect(),
            vec![1, 4],
        ).unwrap();

        // Should not panic — embedding scale applied in forward
        let logits = model.forward(&input).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 100]);
    }

    #[test]
    fn test_falcon_tiny_forward() {
        // Build a tiny Falcon model manually
        let d = 64;
        let num_heads = 4;
        let num_kv = 2;
        let hidden = 256;
        let n_layers = 1;
        let vocab = 100;
        let config = ModelConfig {
            dim: d,
            hidden_dim: hidden,
            n_layers,
            n_heads: num_heads,
            n_kv_heads: Some(num_kv),
            vocab_size: vocab,
            norm_eps: 1e-5,
            max_seq_len: 32,
            use_bias: Some(true),
            position_encoding: PositionEncoding::ALiBi,
            causal: true,
            rope_theta: 10000.0,
            parallel_residual: Some(true),
            ..tiny_config()
        };

        // Create weights dict for from_pretrained_falcon
        let head_dim = d / num_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv * head_dim;
        let qkv_total = q_dim + 2 * kv_dim;

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert("token_embedding.weight".into(), Tensor::randn(vec![vocab, d]));
        weights.insert("norm.weight".into(), Tensor::ones(vec![d]));
        weights.insert("norm.bias".into(), Tensor::zeros(vec![d]));
        weights.insert("output.weight".into(), Tensor::randn(vec![vocab, d]));

        for i in 0..n_layers {
            weights.insert(format!("layers.{}.attention.qkv.weight", i), Tensor::randn(vec![qkv_total, d]));
            weights.insert(format!("layers.{}.attention.qkv.bias", i), Tensor::zeros(vec![qkv_total]));
            weights.insert(format!("layers.{}.attention.out_proj.weight", i), Tensor::randn(vec![d, d]));
            weights.insert(format!("layers.{}.attention.out_proj.bias", i), Tensor::zeros(vec![d]));
            weights.insert(format!("layers.{}.feed_forward.up_proj.weight", i), Tensor::randn(vec![hidden, d]));
            weights.insert(format!("layers.{}.feed_forward.up_proj.bias", i), Tensor::zeros(vec![hidden]));
            weights.insert(format!("layers.{}.feed_forward.down_proj.weight", i), Tensor::randn(vec![d, hidden]));
            weights.insert(format!("layers.{}.feed_forward.down_proj.bias", i), Tensor::zeros(vec![d]));
            weights.insert(format!("layers.{}.attention_norm.weight", i), Tensor::ones(vec![d]));
            weights.insert(format!("layers.{}.attention_norm.bias", i), Tensor::zeros(vec![d]));
            weights.insert(format!("layers.{}.ffn_norm.weight", i), Tensor::ones(vec![d]));
            weights.insert(format!("layers.{}.ffn_norm.bias", i), Tensor::zeros(vec![d]));
        }

        let model = LlmModel::from_pretrained_falcon(&config, weights).unwrap();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let logits = model.forward(&input).unwrap();
        assert_eq!(logits.shape(), &[1, 4, vocab]);
    }

    #[test]
    fn test_mixtral_tiny_forward() {
        let d = 64;
        let hidden = 128;
        let n_layers = 1;
        let vocab = 100;
        let num_experts = 4;
        let num_experts_per_tok = 2;
        let config = ModelConfig {
            dim: d,
            hidden_dim: hidden,
            n_layers,
            n_heads: 4,
            n_kv_heads: Some(2),
            vocab_size: vocab,
            norm_eps: 1e-5,
            max_seq_len: 32,
            use_bias: Some(false),
            position_encoding: PositionEncoding::RoPE,
            causal: true,
            rope_theta: 10000.0,
            num_local_experts: Some(num_experts),
            num_experts_per_tok: Some(num_experts_per_tok),
            ..tiny_config()
        };

        let kv_dim = 2 * (d / 4); // n_kv_heads * head_dim
        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert("token_embedding.weight".into(), Tensor::randn(vec![vocab, d]));
        weights.insert("norm.weight".into(), Tensor::ones(vec![d]));
        weights.insert("output.weight".into(), Tensor::randn(vec![vocab, d]));

        for i in 0..n_layers {
            weights.insert(format!("layers.{}.attention.q_proj.weight", i), Tensor::randn(vec![d, d]));
            weights.insert(format!("layers.{}.attention.k_proj.weight", i), Tensor::randn(vec![kv_dim, d]));
            weights.insert(format!("layers.{}.attention.v_proj.weight", i), Tensor::randn(vec![kv_dim, d]));
            weights.insert(format!("layers.{}.attention.out_proj.weight", i), Tensor::randn(vec![d, d]));
            weights.insert(format!("layers.{}.attention_norm.weight", i), Tensor::ones(vec![d]));
            weights.insert(format!("layers.{}.ffn_norm.weight", i), Tensor::ones(vec![d]));
            // MoE gate
            weights.insert(format!("layers.{}.moe.gate.weight", i), Tensor::randn(vec![num_experts, d]));
            // Expert FFNs
            for j in 0..num_experts {
                weights.insert(format!("layers.{}.moe.experts.{}.gate_proj.weight", i, j), Tensor::randn(vec![hidden, d]));
                weights.insert(format!("layers.{}.moe.experts.{}.up_proj.weight", i, j), Tensor::randn(vec![hidden, d]));
                weights.insert(format!("layers.{}.moe.experts.{}.down_proj.weight", i, j), Tensor::randn(vec![d, hidden]));
            }
        }

        let model = LlmModel::from_pretrained_mixtral(&config, weights).unwrap();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let logits = model.forward(&input).unwrap();
        assert_eq!(logits.shape(), &[1, 4, vocab]);
    }

    #[test]
    fn test_qwen2_attention_bias_config() {
        // Verify that a config with attention_bias=true can be created and model runs
        let config = ModelConfig {
            attention_bias: Some(true),
            position_encoding: PositionEncoding::RoPE,
            ..tiny_config()
        };
        // Just verify config is valid; actual bias loading tested via from_pretrained
        assert!(config.validate().is_ok());
        assert_eq!(config.attention_bias, Some(true));
    }

    #[test]
    fn test_gemma3_tiny_forward() {
        // Build a tiny Gemma 3 model with decoupled head_dim and sliding window pattern
        let d = 64;
        let num_heads = 4;
        let num_kv = 2;
        let head_dim = 32; // decoupled: 64/4=16 != 32
        let hidden = 256;
        let n_layers = 6; // enough to test sliding_window_pattern=3
        let vocab = 100;

        let config = ModelConfig {
            dim: d,
            hidden_dim: hidden,
            n_layers,
            n_heads: num_heads,
            n_kv_heads: Some(num_kv),
            vocab_size: vocab,
            norm_eps: 1e-6,
            max_seq_len: 64,
            use_bias: Some(false),
            position_encoding: PositionEncoding::RoPE,
            causal: true,
            rope_theta: 1000000.0,
            bos_token_id: Some(2),
            eos_token_id: Some(1),
            chat_template: None,
            sliding_window: Some(16),
            attn_logit_cap: None,
            embedding_scale: Some((d as f32).sqrt()),
            rms_norm_offset: Some(1.0),
            attention_bias: None,
            parallel_residual: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            head_dim: Some(head_dim),
            sliding_window_pattern: Some(3),
            query_pre_attn_scalar: Some(32.0),
            rope_local_base_freq: Some(10000.0),
            rope_scaling_factor: Some(8.0),
        };

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv * head_dim;

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        weights.insert("token_embedding.weight".into(), Tensor::randn(vec![vocab, d]));
        weights.insert("norm.weight".into(), Tensor::ones(vec![d]));
        weights.insert("output.weight".into(), Tensor::randn(vec![vocab, d]));

        for i in 0..n_layers {
            weights.insert(format!("layers.{}.attention.q_proj.weight", i), Tensor::randn(vec![q_dim, d]));
            weights.insert(format!("layers.{}.attention.k_proj.weight", i), Tensor::randn(vec![kv_dim, d]));
            weights.insert(format!("layers.{}.attention.v_proj.weight", i), Tensor::randn(vec![kv_dim, d]));
            weights.insert(format!("layers.{}.attention.out_proj.weight", i), Tensor::randn(vec![d, q_dim]));
            // QK norms
            weights.insert(format!("layers.{}.attention.q_norm.weight", i), Tensor::ones(vec![head_dim]));
            weights.insert(format!("layers.{}.attention.k_norm.weight", i), Tensor::ones(vec![head_dim]));
            // FFN
            weights.insert(format!("layers.{}.feed_forward.up_proj.weight", i), Tensor::randn(vec![hidden, d]));
            weights.insert(format!("layers.{}.feed_forward.gate_proj.weight", i), Tensor::randn(vec![hidden, d]));
            weights.insert(format!("layers.{}.feed_forward.down_proj.weight", i), Tensor::randn(vec![d, hidden]));
            // 4 norms (sandwich norm)
            weights.insert(format!("layers.{}.attention_norm.weight", i), Tensor::ones(vec![d]));
            weights.insert(format!("layers.{}.post_attention_norm.weight", i), Tensor::ones(vec![d]));
            weights.insert(format!("layers.{}.ffn_norm.weight", i), Tensor::ones(vec![d]));
            weights.insert(format!("layers.{}.post_ffn_norm.weight", i), Tensor::ones(vec![d]));
        }

        let model = LlmModel::from_pretrained_gemma3(&config, weights).unwrap();

        // Verify model structure
        assert_eq!(model.layers.len(), n_layers);
        assert_eq!(model.config.head_dim, Some(head_dim));

        // Forward pass
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let logits = model.forward(&input).unwrap();
        assert_eq!(logits.shape(), &[1, 4, vocab]);
    }

    // ==================== OptProfile integration tests ====================

    #[test]
    fn test_set_optimization_profile_baseline() {
        use rustml_core::OptProfile;
        let config = tiny_config();
        let mut model = LlmModel::new(&config).unwrap();
        model.set_optimization_profile(OptProfile::Baseline);

        // Verify model still produces valid output with baseline profile
        let input = Tensor::from_vec(
            (0..4).map(|i| (i % 100) as f32).collect(),
            vec![1, 4],
        ).unwrap();
        let logits = model.forward(&input).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 100]);
        let flat = logits.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_set_optimization_profile_baseline_with_cache() {
        use rustml_core::OptProfile;
        let config = tiny_config();
        let mut model = LlmModel::new(&config).unwrap();
        model.set_optimization_profile(OptProfile::Baseline);

        let mut cache = KVCache::new(
            config.n_layers, config.max_seq_len,
            model.head_dim(), model.num_kv_heads(),
        );

        // Prefill
        let input = Tensor::from_vec(
            (0..4).map(|i| (i % 100) as f32).collect(),
            vec![1, 4],
        ).unwrap();
        let logits = model.forward_with_cache(&input, &mut cache).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 100]);
        cache.advance(4);

        // Decode
        let input = Tensor::from_vec(vec![5.0], vec![1, 1]).unwrap();
        let logits = model.forward_with_cache(&input, &mut cache).unwrap();
        assert_eq!(logits.shape(), &[1, 1, 100]);
        let flat = logits.as_slice_f32().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_optimization_profiles_produce_same_greedy_output() {
        use rustml_core::OptProfile;
        let config = tiny_config();

        let input = Tensor::from_vec(
            (0..4).map(|i| (i % 100) as f32).collect(),
            vec![1, 4],
        ).unwrap();

        // Optimized profile
        let mut model_opt = LlmModel::new(&config).unwrap();
        model_opt.set_optimization_profile(OptProfile::Optimized);
        let mut cache_opt = KVCache::new(
            config.n_layers, config.max_seq_len,
            model_opt.head_dim(), model_opt.num_kv_heads(),
        );
        let logits_opt = model_opt.forward_with_cache(&input, &mut cache_opt).unwrap();

        // Baseline profile (same model instance, just toggle)
        model_opt.set_optimization_profile(OptProfile::Baseline);
        let mut cache_base = KVCache::new(
            config.n_layers, config.max_seq_len,
            model_opt.head_dim(), model_opt.num_kv_heads(),
        );
        let logits_base = model_opt.forward_with_cache(&input, &mut cache_base).unwrap();

        let d1 = logits_opt.as_slice_f32().unwrap();
        let d2 = logits_base.as_slice_f32().unwrap();
        assert_eq!(d1.len(), d2.len());
        for i in 0..d1.len() {
            assert!((d1[i] - d2[i]).abs() < 1e-3,
                "profile mismatch at {}: opt={} base={}", i, d1[i], d2[i]);
        }
    }
}
