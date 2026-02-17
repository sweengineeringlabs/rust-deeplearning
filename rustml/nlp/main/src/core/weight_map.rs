//! Weight mapping from HuggingFace tensor names to internal LlmModel names.
//!
//! Provides pre-built mappings for Llama-2 and GPT-2 architectures, and a
//! generic `WeightMap` that can be used for custom mappings.

use crate::api::error::{NlpError, NlpResult};
use rustml_core::Tensor;
use std::collections::HashMap;

/// Maps HuggingFace tensor names to internal model names.
pub struct WeightMap {
    mapping: HashMap<String, String>,
}

impl WeightMap {
    /// Construct from an existing mapping.
    pub fn from_mapping(mapping: HashMap<String, String>) -> Self {
        Self { mapping }
    }

    /// Build a weight mapping for Llama-2 style models.
    ///
    /// Maps:
    /// - `model.embed_tokens.weight` → `token_embedding.weight`
    /// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` → `layers.{i}.attention.{q,k,v,out}_proj.weight`
    /// - `model.layers.{i}.input_layernorm.weight` → `layers.{i}.attention_norm.weight`
    /// - `model.layers.{i}.post_attention_layernorm.weight` → `layers.{i}.ffn_norm.weight`
    /// - `model.layers.{i}.mlp.{up,down,gate}_proj.weight` → `layers.{i}.feed_forward.{up,down,gate}_proj.weight`
    /// - `model.norm.weight` → `norm.weight`
    /// - `lm_head.weight` → `output.weight`
    pub fn llama2(n_layers: usize) -> Self {
        let mut mapping = HashMap::new();

        mapping.insert(
            "model.embed_tokens.weight".into(),
            "token_embedding.weight".into(),
        );
        mapping.insert("model.norm.weight".into(), "norm.weight".into());
        mapping.insert("lm_head.weight".into(), "output.weight".into());

        for i in 0..n_layers {
            mapping.insert(
                format!("model.layers.{}.self_attn.q_proj.weight", i),
                format!("layers.{}.attention.q_proj.weight", i),
            );
            mapping.insert(
                format!("model.layers.{}.self_attn.k_proj.weight", i),
                format!("layers.{}.attention.k_proj.weight", i),
            );
            mapping.insert(
                format!("model.layers.{}.self_attn.v_proj.weight", i),
                format!("layers.{}.attention.v_proj.weight", i),
            );
            mapping.insert(
                format!("model.layers.{}.self_attn.o_proj.weight", i),
                format!("layers.{}.attention.out_proj.weight", i),
            );
            mapping.insert(
                format!("model.layers.{}.input_layernorm.weight", i),
                format!("layers.{}.attention_norm.weight", i),
            );
            mapping.insert(
                format!("model.layers.{}.post_attention_layernorm.weight", i),
                format!("layers.{}.ffn_norm.weight", i),
            );
            mapping.insert(
                format!("model.layers.{}.mlp.up_proj.weight", i),
                format!("layers.{}.feed_forward.up_proj.weight", i),
            );
            mapping.insert(
                format!("model.layers.{}.mlp.down_proj.weight", i),
                format!("layers.{}.feed_forward.down_proj.weight", i),
            );
            mapping.insert(
                format!("model.layers.{}.mlp.gate_proj.weight", i),
                format!("layers.{}.feed_forward.gate_proj.weight", i),
            );
        }

        Self { mapping }
    }

    /// Build a weight mapping for GPT-2 style models.
    ///
    /// Maps HF GPT-2 names (with `transformer.` prefix stripped) to internal names:
    /// - `wte.weight` → `token_embedding.weight`
    /// - `wpe.weight` → `pos_embedding.weight`
    /// - `ln_f.{weight,bias}` → `norm.{weight,bias}`
    /// - `h.{i}.ln_1.*` → `layers.{i}.attention_norm.*`
    /// - `h.{i}.ln_2.*` → `layers.{i}.ffn_norm.*`
    /// - `h.{i}.attn.c_attn.*` → `layers.{i}.attention.c_attn.*` (fused QKV)
    /// - `h.{i}.attn.c_proj.*` → `layers.{i}.attention.out_proj.*`
    /// - `h.{i}.mlp.c_fc.*` → `layers.{i}.feed_forward.up_proj.*`
    /// - `h.{i}.mlp.c_proj.*` → `layers.{i}.feed_forward.down_proj.*`
    pub fn gpt2(n_layers: usize) -> Self {
        let mut mapping = HashMap::new();

        mapping.insert("wte.weight".into(), "token_embedding.weight".into());
        mapping.insert("wpe.weight".into(), "pos_embedding.weight".into());
        mapping.insert("ln_f.weight".into(), "norm.weight".into());
        mapping.insert("ln_f.bias".into(), "norm.bias".into());

        for i in 0..n_layers {
            for suffix in &["weight", "bias"] {
                mapping.insert(
                    format!("h.{}.ln_1.{}", i, suffix),
                    format!("layers.{}.attention_norm.{}", i, suffix),
                );
                mapping.insert(
                    format!("h.{}.ln_2.{}", i, suffix),
                    format!("layers.{}.ffn_norm.{}", i, suffix),
                );
                mapping.insert(
                    format!("h.{}.attn.c_attn.{}", i, suffix),
                    format!("layers.{}.attention.c_attn.{}", i, suffix),
                );
                mapping.insert(
                    format!("h.{}.attn.c_proj.{}", i, suffix),
                    format!("layers.{}.attention.out_proj.{}", i, suffix),
                );
                mapping.insert(
                    format!("h.{}.mlp.c_fc.{}", i, suffix),
                    format!("layers.{}.feed_forward.up_proj.{}", i, suffix),
                );
                mapping.insert(
                    format!("h.{}.mlp.c_proj.{}", i, suffix),
                    format!("layers.{}.feed_forward.down_proj.{}", i, suffix),
                );
            }
        }

        Self { mapping }
    }

    /// Build a weight mapping for Llama-2 style models with attention bias (Qwen-2).
    ///
    /// Extends `llama2()` with Q/K/V/O `.bias` entries.
    pub fn llama2_with_attn_bias(n_layers: usize) -> Self {
        let mut base = Self::llama2(n_layers);
        for i in 0..n_layers {
            base.mapping.insert(
                format!("model.layers.{}.self_attn.q_proj.bias", i),
                format!("layers.{}.attention.q_proj.bias", i),
            );
            base.mapping.insert(
                format!("model.layers.{}.self_attn.k_proj.bias", i),
                format!("layers.{}.attention.k_proj.bias", i),
            );
            base.mapping.insert(
                format!("model.layers.{}.self_attn.v_proj.bias", i),
                format!("layers.{}.attention.v_proj.bias", i),
            );
            base.mapping.insert(
                format!("model.layers.{}.self_attn.o_proj.bias", i),
                format!("layers.{}.attention.out_proj.bias", i),
            );
        }
        base
    }

    /// Build a weight mapping for Falcon models.
    ///
    /// Maps fused QKV, LayerNorm with bias, and parallel residual architecture.
    pub fn falcon(n_layers: usize) -> Self {
        let mut mapping = HashMap::new();

        mapping.insert(
            "transformer.word_embeddings.weight".into(),
            "token_embedding.weight".into(),
        );
        mapping.insert("transformer.ln_f.weight".into(), "norm.weight".into());
        mapping.insert("transformer.ln_f.bias".into(), "norm.bias".into());
        mapping.insert("lm_head.weight".into(), "output.weight".into());

        for i in 0..n_layers {
            // Fused QKV
            for suffix in &["weight", "bias"] {
                mapping.insert(
                    format!("transformer.h.{}.self_attention.query_key_value.{}", i, suffix),
                    format!("layers.{}.attention.qkv.{}", i, suffix),
                );
                mapping.insert(
                    format!("transformer.h.{}.self_attention.dense.{}", i, suffix),
                    format!("layers.{}.attention.out_proj.{}", i, suffix),
                );
                mapping.insert(
                    format!("transformer.h.{}.mlp.dense_h_to_4h.{}", i, suffix),
                    format!("layers.{}.feed_forward.up_proj.{}", i, suffix),
                );
                mapping.insert(
                    format!("transformer.h.{}.mlp.dense_4h_to_h.{}", i, suffix),
                    format!("layers.{}.feed_forward.down_proj.{}", i, suffix),
                );
            }

            // Attention norm (input_layernorm / ln_attn)
            for suffix in &["weight", "bias"] {
                mapping.insert(
                    format!("transformer.h.{}.input_layernorm.{}", i, suffix),
                    format!("layers.{}.attention_norm.{}", i, suffix),
                );
                mapping.insert(
                    format!("transformer.h.{}.ln_attn.{}", i, suffix),
                    format!("layers.{}.attention_norm.{}", i, suffix),
                );
            }

            // FFN norm (post_attention_layernorm / ln_mlp)
            for suffix in &["weight", "bias"] {
                mapping.insert(
                    format!("transformer.h.{}.post_attention_layernorm.{}", i, suffix),
                    format!("layers.{}.ffn_norm.{}", i, suffix),
                );
                mapping.insert(
                    format!("transformer.h.{}.ln_mlp.{}", i, suffix),
                    format!("layers.{}.ffn_norm.{}", i, suffix),
                );
            }
        }

        Self { mapping }
    }

    /// Build a weight mapping for Mixtral MoE models.
    ///
    /// Attention follows Llama-2 style. MoE replaces the MLP with a router gate
    /// and N expert SwiGLU FFNs.
    pub fn mixtral(n_layers: usize, num_experts: usize) -> Self {
        let mut base = Self::llama2(n_layers);

        for i in 0..n_layers {
            // Remove Llama-2 MLP mappings for this layer
            base.mapping.remove(&format!("model.layers.{}.mlp.up_proj.weight", i));
            base.mapping.remove(&format!("model.layers.{}.mlp.down_proj.weight", i));
            base.mapping.remove(&format!("model.layers.{}.mlp.gate_proj.weight", i));

            // Router gate
            base.mapping.insert(
                format!("model.layers.{}.block_sparse_moe.gate.weight", i),
                format!("layers.{}.moe.gate.weight", i),
            );

            // Expert FFNs
            for j in 0..num_experts {
                base.mapping.insert(
                    format!("model.layers.{}.block_sparse_moe.experts.{}.w1.weight", i, j),
                    format!("layers.{}.moe.experts.{}.gate_proj.weight", i, j),
                );
                base.mapping.insert(
                    format!("model.layers.{}.block_sparse_moe.experts.{}.w2.weight", i, j),
                    format!("layers.{}.moe.experts.{}.down_proj.weight", i, j),
                );
                base.mapping.insert(
                    format!("model.layers.{}.block_sparse_moe.experts.{}.w3.weight", i, j),
                    format!("layers.{}.moe.experts.{}.up_proj.weight", i, j),
                );
            }
        }

        base
    }

    /// Remap HuggingFace weight names to internal names.
    /// Unmapped keys are skipped with a warning to stderr.
    pub fn remap(&self, hf_weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        let mut remapped = HashMap::new();

        for (hf_name, tensor) in hf_weights {
            // Strip "transformer." prefix if present (HF GPT-2 uses it)
            let stripped = hf_name
                .strip_prefix("transformer.")
                .unwrap_or(&hf_name);

            if let Some(internal_name) = self.mapping.get(stripped) {
                remapped.insert(internal_name.clone(), tensor);
            } else if let Some(internal_name) = self.mapping.get(&hf_name) {
                remapped.insert(internal_name.clone(), tensor);
            } else {
                eprintln!("WeightMap: skipping unmapped tensor '{}'", hf_name);
            }
        }

        remapped
    }

    /// Validate that all required weights are present.
    pub fn validate(&self, weights: &HashMap<String, Tensor>) -> NlpResult<()> {
        let mut missing = Vec::new();
        for internal_name in self.mapping.values() {
            if !weights.contains_key(internal_name) {
                missing.push(internal_name.clone());
            }
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(NlpError::ModelError(format!(
                "Missing required weights: {}",
                missing.join(", ")
            )))
        }
    }

    /// Number of mappings.
    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    /// Whether the mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama2_weight_map() {
        let wm = WeightMap::llama2(2);
        // 3 global + 9*2 per-layer = 21
        assert_eq!(wm.len(), 3 + 9 * 2);

        let mut hf_weights = HashMap::new();
        hf_weights.insert(
            "model.embed_tokens.weight".into(),
            Tensor::randn(vec![100, 64]),
        );
        hf_weights.insert(
            "model.norm.weight".into(),
            Tensor::randn(vec![64]),
        );
        hf_weights.insert(
            "lm_head.weight".into(),
            Tensor::randn(vec![100, 64]),
        );

        let remapped = wm.remap(hf_weights);
        assert!(remapped.contains_key("token_embedding.weight"));
        assert!(remapped.contains_key("norm.weight"));
        assert!(remapped.contains_key("output.weight"));
    }

    #[test]
    fn test_gpt2_weight_map() {
        let wm = WeightMap::gpt2(2);
        // 4 global + 12*2 per-layer = 28
        assert_eq!(wm.len(), 4 + 12 * 2);

        let mut hf_weights = HashMap::new();
        hf_weights.insert(
            "transformer.wte.weight".into(),
            Tensor::randn(vec![100, 64]),
        );
        hf_weights.insert(
            "transformer.wpe.weight".into(),
            Tensor::randn(vec![32, 64]),
        );

        let remapped = wm.remap(hf_weights);
        assert!(remapped.contains_key("token_embedding.weight"));
        assert!(remapped.contains_key("pos_embedding.weight"));
    }

    #[test]
    fn test_falcon_weight_map() {
        let wm = WeightMap::falcon(2);
        // 4 global + per-layer: 8 (qkv/dense/h_to_4h/4h_to_h * w+b) + 4 (ln_attn/ln_mlp * w+b)
        //   + 4 (input_layernorm/post_attention_layernorm * w+b) = 16 per layer
        // But ln_attn maps to same target as input_layernorm so count source keys:
        // 4 global + 16*2 = 36
        assert_eq!(wm.len(), 4 + 16 * 2);
    }

    #[test]
    fn test_mixtral_weight_map() {
        let wm = WeightMap::mixtral(2, 4);
        // Base llama2: 3 global + 9*2 per-layer = 21
        // Minus 3 MLP per layer (up/down/gate) = 21 - 6 = 15
        // Plus per layer: 1 gate + 4 experts * 3 weights = 13
        // Total: 15 + 13*2 = 41
        assert_eq!(wm.len(), 15 + 13 * 2);
    }

    #[test]
    fn test_llama2_with_attn_bias_weight_map() {
        let wm = WeightMap::llama2_with_attn_bias(2);
        // Base llama2: 3 + 9*2 = 21
        // Plus 4 bias entries per layer = 21 + 4*2 = 29
        assert_eq!(wm.len(), 21 + 4 * 2);
    }

    #[test]
    fn test_custom_weight_map() {
        let mut mapping = HashMap::new();
        mapping.insert("src.a".into(), "dst.a".into());
        mapping.insert("src.b".into(), "dst.b".into());

        let wm = WeightMap::from_mapping(mapping);
        assert_eq!(wm.len(), 2);
    }
}
