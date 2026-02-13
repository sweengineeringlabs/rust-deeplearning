use std::collections::HashMap;
use crate::core::tensor::Tensor;
use crate::error::{LLMForgeError, Result};

/// Maps HuggingFace tensor names to LLMForge internal names.
pub struct WeightMap {
    mapping: HashMap<String, String>,
}

impl WeightMap {
    /// Construct a WeightMap from an existing mapping HashMap.
    pub fn from_mapping(mapping: HashMap<String, String>) -> Self {
        Self { mapping }
    }

    /// Build a weight mapping for Llama-2 style models.
    ///
    /// Produces 3 global mappings + 8 per layer:
    /// - `model.embed_tokens.weight` → `token_embedding.weight`
    /// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` → `layers.{i}.attention.{q,k,v,out}_proj.weight`
    /// - `model.layers.{i}.input_layernorm.weight` → `layers.{i}.attention_norm.weight`
    /// - `model.layers.{i}.post_attention_layernorm.weight` → `layers.{i}.ffn_norm.weight`
    /// - `model.layers.{i}.mlp.{up,down}_proj.weight` → `layers.{i}.feed_forward.{up,down}_proj.weight`
    /// - `model.norm.weight` → `norm.weight`
    /// - `lm_head.weight` → `output.weight`
    pub fn llama2(n_layers: usize) -> Self {
        let mut mapping = HashMap::new();

        // Global mappings
        mapping.insert(
            "model.embed_tokens.weight".to_string(),
            "token_embedding.weight".to_string(),
        );
        mapping.insert(
            "model.norm.weight".to_string(),
            "norm.weight".to_string(),
        );
        mapping.insert(
            "lm_head.weight".to_string(),
            "output.weight".to_string(),
        );

        // Per-layer mappings
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

    /// Remap HuggingFace weight names to internal names.
    /// Unmapped keys are skipped with a warning printed to stderr.
    pub fn remap(&self, hf_weights: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let mut remapped = HashMap::new();

        for (hf_name, tensor) in hf_weights {
            if let Some(internal_name) = self.mapping.get(&hf_name) {
                remapped.insert(internal_name.clone(), tensor);
            } else {
                eprintln!("WeightMap: skipping unmapped tensor '{}'", hf_name);
            }
        }

        Ok(remapped)
    }

    /// Validate that all required weights are present for a Llama-2 model.
    pub fn validate(&self, weights: &HashMap<String, Tensor>, n_layers: usize) -> Result<()> {
        let mut missing = Vec::new();

        // Check global keys
        for key in &["token_embedding.weight", "norm.weight", "output.weight"] {
            if !weights.contains_key(*key) {
                missing.push(key.to_string());
            }
        }

        // Check per-layer keys
        for i in 0..n_layers {
            let layer_keys = [
                format!("layers.{}.attention.q_proj.weight", i),
                format!("layers.{}.attention.k_proj.weight", i),
                format!("layers.{}.attention.v_proj.weight", i),
                format!("layers.{}.attention.out_proj.weight", i),
                format!("layers.{}.attention_norm.weight", i),
                format!("layers.{}.ffn_norm.weight", i),
                format!("layers.{}.feed_forward.up_proj.weight", i),
                format!("layers.{}.feed_forward.down_proj.weight", i),
            ];
            for key in &layer_keys {
                if !weights.contains_key(key) {
                    missing.push(key.clone());
                }
            }
        }

        if missing.is_empty() {
            Ok(())
        } else {
            Err(LLMForgeError::NotImplemented(
                format!("Missing required weights: {}", missing.join(", "))
            ))
        }
    }

    /// Build a weight mapping for GPT-2 style models.
    ///
    /// GPT-2 HF naming conventions:
    /// - `wte.weight` → `token_embedding.weight`
    /// - `wpe.weight` → `pos_embedding.weight`
    /// - `ln_f.weight/bias` → `norm.weight/bias`
    /// - `h.{i}.ln_1.weight/bias` → `layers.{i}.attention_norm.weight/bias`
    /// - `h.{i}.ln_2.weight/bias` → `layers.{i}.ffn_norm.weight/bias`
    /// - `h.{i}.attn.c_attn.weight/bias` → `layers.{i}.attention.c_attn.weight/bias` (fused QKV)
    /// - `h.{i}.attn.c_proj.weight/bias` → `layers.{i}.attention.out_proj.weight/bias`
    /// - `h.{i}.mlp.c_fc.weight/bias` → `layers.{i}.feed_forward.up_proj.weight/bias`
    /// - `h.{i}.mlp.c_proj.weight/bias` → `layers.{i}.feed_forward.down_proj.weight/bias`
    pub fn gpt2(n_layers: usize) -> Self {
        let mut mapping = HashMap::new();

        // Global mappings
        mapping.insert("wte.weight".to_string(), "token_embedding.weight".to_string());
        mapping.insert("wpe.weight".to_string(), "pos_embedding.weight".to_string());
        mapping.insert("ln_f.weight".to_string(), "norm.weight".to_string());
        mapping.insert("ln_f.bias".to_string(), "norm.bias".to_string());

        // Per-layer mappings
        for i in 0..n_layers {
            // Attention norms
            mapping.insert(
                format!("h.{}.ln_1.weight", i),
                format!("layers.{}.attention_norm.weight", i),
            );
            mapping.insert(
                format!("h.{}.ln_1.bias", i),
                format!("layers.{}.attention_norm.bias", i),
            );
            mapping.insert(
                format!("h.{}.ln_2.weight", i),
                format!("layers.{}.ffn_norm.weight", i),
            );
            mapping.insert(
                format!("h.{}.ln_2.bias", i),
                format!("layers.{}.ffn_norm.bias", i),
            );

            // Fused QKV attention (split later)
            mapping.insert(
                format!("h.{}.attn.c_attn.weight", i),
                format!("layers.{}.attention.c_attn.weight", i),
            );
            mapping.insert(
                format!("h.{}.attn.c_attn.bias", i),
                format!("layers.{}.attention.c_attn.bias", i),
            );

            // Output projection
            mapping.insert(
                format!("h.{}.attn.c_proj.weight", i),
                format!("layers.{}.attention.out_proj.weight", i),
            );
            mapping.insert(
                format!("h.{}.attn.c_proj.bias", i),
                format!("layers.{}.attention.out_proj.bias", i),
            );

            // FFN
            mapping.insert(
                format!("h.{}.mlp.c_fc.weight", i),
                format!("layers.{}.feed_forward.up_proj.weight", i),
            );
            mapping.insert(
                format!("h.{}.mlp.c_fc.bias", i),
                format!("layers.{}.feed_forward.up_proj.bias", i),
            );
            mapping.insert(
                format!("h.{}.mlp.c_proj.weight", i),
                format!("layers.{}.feed_forward.down_proj.weight", i),
            );
            mapping.insert(
                format!("h.{}.mlp.c_proj.bias", i),
                format!("layers.{}.feed_forward.down_proj.bias", i),
            );
        }

        Self { mapping }
    }

    /// Returns the number of mappings.
    pub fn len(&self) -> usize {
        self.mapping.len()
    }
}
