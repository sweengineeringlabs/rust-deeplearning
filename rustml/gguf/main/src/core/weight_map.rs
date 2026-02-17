use std::collections::HashMap;

/// Maps tensor names from GGUF/HuggingFace conventions to internal names.
pub struct WeightMap {
    mapping: HashMap<String, String>,
}

impl WeightMap {
    pub fn from_mapping(mapping: HashMap<String, String>) -> Self {
        Self { mapping }
    }

    /// Remap tensor names from source names to internal names.
    /// Tensors without a mapping are passed through with their original name.
    pub fn remap<V>(&self, tensors: HashMap<String, V>) -> HashMap<String, V> {
        tensors.into_iter()
            .map(|(name, tensor)| {
                let mapped_name = self.mapping.get(&name).cloned().unwrap_or(name);
                (mapped_name, tensor)
            })
            .collect()
    }
}

/// Build a weight mapping for GGUF Llama-style tensor names.
pub fn gguf_llama_weight_map(n_layers: usize) -> WeightMap {
    let mut mapping = HashMap::new();

    mapping.insert("token_embd.weight".to_string(), "token_embedding.weight".to_string());
    mapping.insert("output_norm.weight".to_string(), "norm.weight".to_string());
    mapping.insert("output.weight".to_string(), "output.weight".to_string());

    for i in 0..n_layers {
        mapping.insert(
            format!("blk.{}.attn_q.weight", i),
            format!("layers.{}.attention.q_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_k.weight", i),
            format!("layers.{}.attention.k_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_v.weight", i),
            format!("layers.{}.attention.v_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_output.weight", i),
            format!("layers.{}.attention.out_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_norm.weight", i),
            format!("layers.{}.attention_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_norm.weight", i),
            format!("layers.{}.ffn_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_up.weight", i),
            format!("layers.{}.feed_forward.up_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_down.weight", i),
            format!("layers.{}.feed_forward.down_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_gate.weight", i),
            format!("layers.{}.feed_forward.gate_proj.weight", i),
        );
    }

    WeightMap::from_mapping(mapping)
}

/// Build a weight mapping for GGUF Gemma 3 tensor names.
///
/// Extends the Llama mapping with Gemma 3 specifics:
/// - QK normalization: `blk.{i}.attn_q_norm`, `blk.{i}.attn_k_norm`
/// - 4 layer norms (sandwich norm):
///   - `blk.{i}.attn_norm` → pre-attention norm (same as Llama)
///   - `blk.{i}.post_attention_norm` → post-attention norm
///   - `blk.{i}.ffn_norm` → pre-FFN norm (same as Llama)
///   - `blk.{i}.post_ffw_norm` → post-FFN norm
pub fn gguf_gemma3_weight_map(n_layers: usize) -> WeightMap {
    let mut mapping = HashMap::new();

    mapping.insert("token_embd.weight".to_string(), "token_embedding.weight".to_string());
    mapping.insert("output_norm.weight".to_string(), "norm.weight".to_string());
    mapping.insert("output.weight".to_string(), "output.weight".to_string());

    for i in 0..n_layers {
        // Attention projections (same as Llama)
        mapping.insert(
            format!("blk.{}.attn_q.weight", i),
            format!("layers.{}.attention.q_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_k.weight", i),
            format!("layers.{}.attention.k_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_v.weight", i),
            format!("layers.{}.attention.v_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_output.weight", i),
            format!("layers.{}.attention.out_proj.weight", i),
        );

        // QK normalization (Gemma 3)
        mapping.insert(
            format!("blk.{}.attn_q_norm.weight", i),
            format!("layers.{}.attention.q_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_k_norm.weight", i),
            format!("layers.{}.attention.k_norm.weight", i),
        );

        // 4 layer norms (Gemma 3 sandwich norm)
        mapping.insert(
            format!("blk.{}.attn_norm.weight", i),
            format!("layers.{}.attention_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.post_attention_norm.weight", i),
            format!("layers.{}.post_attention_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_norm.weight", i),
            format!("layers.{}.ffn_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.post_ffw_norm.weight", i),
            format!("layers.{}.post_ffn_norm.weight", i),
        );

        // MLP (same as Llama)
        mapping.insert(
            format!("blk.{}.ffn_up.weight", i),
            format!("layers.{}.feed_forward.up_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_down.weight", i),
            format!("layers.{}.feed_forward.down_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_gate.weight", i),
            format!("layers.{}.feed_forward.gate_proj.weight", i),
        );
    }

    WeightMap::from_mapping(mapping)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_llama_weight_map() {
        let wm = gguf_llama_weight_map(2);

        let mut tensors = HashMap::new();
        tensors.insert("token_embd.weight".to_string(), "tok");
        tensors.insert("blk.0.attn_q.weight".to_string(), "q0");
        tensors.insert("blk.1.ffn_gate.weight".to_string(), "gate1");

        let remapped = wm.remap(tensors);
        assert!(remapped.contains_key("token_embedding.weight"));
        assert!(remapped.contains_key("layers.0.attention.q_proj.weight"));
        assert!(remapped.contains_key("layers.1.feed_forward.gate_proj.weight"));
    }

    #[test]
    fn test_gguf_gemma3_weight_map() {
        let wm = gguf_gemma3_weight_map(2);

        let mut tensors = HashMap::new();
        tensors.insert("token_embd.weight".to_string(), "tok");
        tensors.insert("blk.0.attn_q_norm.weight".to_string(), "qn0");
        tensors.insert("blk.0.attn_k_norm.weight".to_string(), "kn0");
        tensors.insert("blk.0.post_attention_norm.weight".to_string(), "pan0");
        tensors.insert("blk.0.post_ffw_norm.weight".to_string(), "pfn0");
        tensors.insert("blk.1.attn_q.weight".to_string(), "q1");

        let remapped = wm.remap(tensors);
        assert!(remapped.contains_key("token_embedding.weight"));
        assert!(remapped.contains_key("layers.0.attention.q_norm.weight"));
        assert!(remapped.contains_key("layers.0.attention.k_norm.weight"));
        assert!(remapped.contains_key("layers.0.post_attention_norm.weight"));
        assert!(remapped.contains_key("layers.0.post_ffn_norm.weight"));
        assert!(remapped.contains_key("layers.1.attention.q_proj.weight"));
    }

    #[test]
    fn test_gguf_gemma3_all_keys() {
        let wm = gguf_gemma3_weight_map(1);

        let mut tensors = HashMap::new();
        for key in [
            "token_embd.weight", "output_norm.weight", "output.weight",
            "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight",
            "blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight",
            "blk.0.ffn_norm.weight", "blk.0.post_ffw_norm.weight",
            "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.0.ffn_gate.weight",
        ] {
            tensors.insert(key.to_string(), key);
        }
        // 3 global + 13 per-layer = 16
        assert_eq!(tensors.len(), 16);

        let remapped = wm.remap(tensors);
        assert!(!remapped.contains_key("token_embd.weight"));
        assert!(remapped.contains_key("token_embedding.weight"));
        assert_eq!(remapped.len(), 16);
    }
}
