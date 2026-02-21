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
        tensors
            .into_iter()
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

    mapping.insert(
        "token_embd.weight".to_string(),
        "token_embedding.weight".to_string(),
    );
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

    mapping.insert(
        "token_embd.weight".to_string(),
        "token_embedding.weight".to_string(),
    );
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

/// Build a weight mapping for GGUF BERT-style tensor names.
///
/// BERT models use different naming conventions from Llama:
/// - Bias on all projections and norms
/// - `attn_output_norm` / `layer_output_norm` instead of `attn_norm` / `ffn_norm`
/// - Learned position embeddings (`position_embd.weight`)
/// - Embedding layer norm (`token_embd_norm.{weight,bias}`)
/// - No gate_proj (GELU FFN, not SwiGLU)
pub fn gguf_bert_weight_map(n_layers: usize) -> WeightMap {
    let mut mapping = HashMap::new();

    // Global tensors
    mapping.insert(
        "token_embd.weight".to_string(),
        "token_embedding.weight".to_string(),
    );
    mapping.insert(
        "position_embd.weight".to_string(),
        "pos_embedding.weight".to_string(),
    );
    mapping.insert(
        "token_embd_norm.weight".to_string(),
        "embd_norm.weight".to_string(),
    );
    mapping.insert(
        "token_embd_norm.bias".to_string(),
        "embd_norm.bias".to_string(),
    );

    for i in 0..n_layers {
        // Attention projections with bias
        for (gguf, internal) in &[
            ("attn_q", "attention.q_proj"),
            ("attn_k", "attention.k_proj"),
            ("attn_v", "attention.v_proj"),
            ("attn_output", "attention.out_proj"),
        ] {
            mapping.insert(
                format!("blk.{}.{}.weight", i, gguf),
                format!("layers.{}.{}.weight", i, internal),
            );
            mapping.insert(
                format!("blk.{}.{}.bias", i, gguf),
                format!("layers.{}.{}.bias", i, internal),
            );
        }

        // Post-attention LayerNorm (BERT: attn_output_norm → attention_norm)
        mapping.insert(
            format!("blk.{}.attn_output_norm.weight", i),
            format!("layers.{}.attention_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_output_norm.bias", i),
            format!("layers.{}.attention_norm.bias", i),
        );

        // FFN projections with bias (no gate_proj)
        mapping.insert(
            format!("blk.{}.ffn_up.weight", i),
            format!("layers.{}.feed_forward.up_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_up.bias", i),
            format!("layers.{}.feed_forward.up_proj.bias", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_down.weight", i),
            format!("layers.{}.feed_forward.down_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_down.bias", i),
            format!("layers.{}.feed_forward.down_proj.bias", i),
        );

        // Post-FFN LayerNorm (BERT: layer_output_norm → ffn_norm)
        mapping.insert(
            format!("blk.{}.layer_output_norm.weight", i),
            format!("layers.{}.ffn_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.layer_output_norm.bias", i),
            format!("layers.{}.ffn_norm.bias", i),
        );
    }

    WeightMap::from_mapping(mapping)
}

/// Build a weight mapping for GGUF Nomic-BERT tensor names.
///
/// Nomic-BERT differs from standard BERT:
/// - Fused QKV (`attn_qkv.weight`, no bias) — splitting happens in model constructor
/// - SwiGLU FFN (`ffn_gate` + `ffn_up` + `ffn_down`, no bias)
/// - RoPE position encoding (no `position_embd`)
/// - No bias on attention projections
/// - Post-norm LayerNorm with bias on norms only
pub fn gguf_nomic_bert_weight_map(n_layers: usize) -> WeightMap {
    let mut mapping = HashMap::new();

    // Global tensors (3)
    mapping.insert(
        "token_embd.weight".to_string(),
        "token_embedding.weight".to_string(),
    );
    mapping.insert(
        "token_embd_norm.weight".to_string(),
        "embd_norm.weight".to_string(),
    );
    mapping.insert(
        "token_embd_norm.bias".to_string(),
        "embd_norm.bias".to_string(),
    );

    for i in 0..n_layers {
        // Fused QKV (no bias)
        mapping.insert(
            format!("blk.{}.attn_qkv.weight", i),
            format!("layers.{}.attention.qkv.weight", i),
        );
        // Output projection (no bias)
        mapping.insert(
            format!("blk.{}.attn_output.weight", i),
            format!("layers.{}.attention.out_proj.weight", i),
        );
        // Post-attention LayerNorm with bias
        mapping.insert(
            format!("blk.{}.attn_output_norm.weight", i),
            format!("layers.{}.attention_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.attn_output_norm.bias", i),
            format!("layers.{}.attention_norm.bias", i),
        );
        // SwiGLU FFN (no bias)
        mapping.insert(
            format!("blk.{}.ffn_gate.weight", i),
            format!("layers.{}.feed_forward.gate_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_up.weight", i),
            format!("layers.{}.feed_forward.up_proj.weight", i),
        );
        mapping.insert(
            format!("blk.{}.ffn_down.weight", i),
            format!("layers.{}.feed_forward.down_proj.weight", i),
        );
        // Post-FFN LayerNorm with bias
        mapping.insert(
            format!("blk.{}.layer_output_norm.weight", i),
            format!("layers.{}.ffn_norm.weight", i),
        );
        mapping.insert(
            format!("blk.{}.layer_output_norm.bias", i),
            format!("layers.{}.ffn_norm.bias", i),
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
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.post_ffw_norm.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_gate.weight",
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

    #[test]
    fn test_gguf_bert_weight_map() {
        let wm = gguf_bert_weight_map(2);

        let mut tensors = HashMap::new();
        tensors.insert("token_embd.weight".to_string(), "tok");
        tensors.insert("blk.0.attn_q.weight".to_string(), "q0");
        tensors.insert("blk.0.attn_q.bias".to_string(), "qb0");
        tensors.insert("blk.0.attn_output_norm.weight".to_string(), "an0");
        tensors.insert("blk.1.layer_output_norm.bias".to_string(), "fnb1");
        tensors.insert("position_embd.weight".to_string(), "pos");
        tensors.insert("token_embd_norm.weight".to_string(), "enw");

        let remapped = wm.remap(tensors);
        assert!(remapped.contains_key("token_embedding.weight"));
        assert!(remapped.contains_key("layers.0.attention.q_proj.weight"));
        assert!(remapped.contains_key("layers.0.attention.q_proj.bias"));
        assert!(remapped.contains_key("layers.0.attention_norm.weight"));
        assert!(remapped.contains_key("layers.1.ffn_norm.bias"));
        assert!(remapped.contains_key("pos_embedding.weight"));
        assert!(remapped.contains_key("embd_norm.weight"));
    }

    #[test]
    fn test_gguf_bert_all_keys() {
        let wm = gguf_bert_weight_map(1);

        let mut tensors = HashMap::new();
        for key in [
            // 4 global tensors
            "token_embd.weight",
            "position_embd.weight",
            "token_embd_norm.weight",
            "token_embd_norm.bias",
            // 16 per-layer tensors (4 attn w+b, 1 attn_norm w+b, 2 ffn w+b, 1 ffn_norm w+b)
            "blk.0.attn_q.weight",
            "blk.0.attn_q.bias",
            "blk.0.attn_k.weight",
            "blk.0.attn_k.bias",
            "blk.0.attn_v.weight",
            "blk.0.attn_v.bias",
            "blk.0.attn_output.weight",
            "blk.0.attn_output.bias",
            "blk.0.attn_output_norm.weight",
            "blk.0.attn_output_norm.bias",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_up.bias",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_down.bias",
            "blk.0.layer_output_norm.weight",
            "blk.0.layer_output_norm.bias",
        ] {
            tensors.insert(key.to_string(), key);
        }
        // 4 global + 16 per-layer = 20
        assert_eq!(tensors.len(), 20);

        let remapped = wm.remap(tensors);
        assert!(!remapped.contains_key("token_embd.weight"));
        assert!(remapped.contains_key("token_embedding.weight"));
        assert!(remapped.contains_key("pos_embedding.weight"));
        assert!(remapped.contains_key("embd_norm.weight"));
        assert!(remapped.contains_key("embd_norm.bias"));
        assert_eq!(remapped.len(), 20);
    }

    #[test]
    fn test_gguf_nomic_bert_weight_map() {
        let wm = gguf_nomic_bert_weight_map(2);

        let mut tensors = HashMap::new();
        tensors.insert("token_embd.weight".to_string(), "tok");
        tensors.insert("blk.0.attn_qkv.weight".to_string(), "qkv0");
        tensors.insert("blk.0.attn_output.weight".to_string(), "out0");
        tensors.insert("blk.1.ffn_gate.weight".to_string(), "gate1");
        tensors.insert("token_embd_norm.weight".to_string(), "enw");

        let remapped = wm.remap(tensors);
        assert!(remapped.contains_key("token_embedding.weight"));
        assert!(remapped.contains_key("layers.0.attention.qkv.weight"));
        assert!(remapped.contains_key("layers.0.attention.out_proj.weight"));
        assert!(remapped.contains_key("layers.1.feed_forward.gate_proj.weight"));
        assert!(remapped.contains_key("embd_norm.weight"));
    }

    #[test]
    fn test_gguf_nomic_bert_all_keys() {
        let wm = gguf_nomic_bert_weight_map(1);

        let mut tensors = HashMap::new();
        for key in [
            // 3 global tensors
            "token_embd.weight",
            "token_embd_norm.weight",
            "token_embd_norm.bias",
            // 9 per-layer tensors
            "blk.0.attn_qkv.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_output_norm.weight",
            "blk.0.attn_output_norm.bias",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.layer_output_norm.weight",
            "blk.0.layer_output_norm.bias",
        ] {
            tensors.insert(key.to_string(), key);
        }
        // 3 global + 9 per-layer = 12
        assert_eq!(tensors.len(), 12);

        let remapped = wm.remap(tensors);
        assert!(!remapped.contains_key("token_embd.weight"));
        assert!(remapped.contains_key("token_embedding.weight"));
        assert!(remapped.contains_key("embd_norm.weight"));
        assert!(remapped.contains_key("embd_norm.bias"));
        assert!(remapped.contains_key("layers.0.attention.qkv.weight"));
        assert!(remapped.contains_key("layers.0.feed_forward.gate_proj.weight"));
        assert_eq!(remapped.len(), 12);
    }
}
