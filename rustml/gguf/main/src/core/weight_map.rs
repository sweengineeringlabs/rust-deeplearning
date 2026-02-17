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
