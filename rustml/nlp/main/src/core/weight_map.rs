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
    fn test_custom_weight_map() {
        let mut mapping = HashMap::new();
        mapping.insert("src.a".into(), "dst.a".into());
        mapping.insert("src.b".into(), "dst.b".into());

        let wm = WeightMap::from_mapping(mapping);
        assert_eq!(wm.len(), 2);
    }
}
