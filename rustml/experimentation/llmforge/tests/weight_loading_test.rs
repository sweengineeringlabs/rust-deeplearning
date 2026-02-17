use std::collections::HashMap;
use llmforge::core::tensor::{Tensor, DType};
use llmforge::loader::WeightMap;
use llmforge::config::{ModelConfig, PositionEncoding};
use llmforge::models::LlmModel;

/// Helper: create a random F32 tensor of the given shape (zero-filled for simplicity).
fn make_tensor(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape)
}

#[test]
fn weight_map_llama2_correct_count() {
    // 3 global + 9 per layer (including gate_proj)
    let n_layers = 4;
    let wmap = WeightMap::llama2(n_layers);
    assert_eq!(wmap.len(), 3 + 9 * n_layers);
}

#[test]
fn weight_map_llama2_single_layer_count() {
    let wmap = WeightMap::llama2(1);
    assert_eq!(wmap.len(), 3 + 9);
}

#[test]
fn weight_map_remap_translates_names() {
    let wmap = WeightMap::llama2(2);

    let mut hf_weights = HashMap::new();
    hf_weights.insert("model.embed_tokens.weight".to_string(), make_tensor(&[100, 32]));
    hf_weights.insert("model.norm.weight".to_string(), make_tensor(&[32]));
    hf_weights.insert("lm_head.weight".to_string(), make_tensor(&[100, 32]));

    for i in 0..2 {
        hf_weights.insert(format!("model.layers.{}.self_attn.q_proj.weight", i), make_tensor(&[32, 32]));
        hf_weights.insert(format!("model.layers.{}.self_attn.k_proj.weight", i), make_tensor(&[32, 32]));
        hf_weights.insert(format!("model.layers.{}.self_attn.v_proj.weight", i), make_tensor(&[32, 32]));
        hf_weights.insert(format!("model.layers.{}.self_attn.o_proj.weight", i), make_tensor(&[32, 32]));
        hf_weights.insert(format!("model.layers.{}.input_layernorm.weight", i), make_tensor(&[32]));
        hf_weights.insert(format!("model.layers.{}.post_attention_layernorm.weight", i), make_tensor(&[32]));
        hf_weights.insert(format!("model.layers.{}.mlp.up_proj.weight", i), make_tensor(&[64, 32]));
        hf_weights.insert(format!("model.layers.{}.mlp.down_proj.weight", i), make_tensor(&[32, 64]));
    }

    let remapped = wmap.remap(hf_weights).unwrap();

    // Check key translations
    assert!(remapped.contains_key("token_embedding.weight"));
    assert!(remapped.contains_key("norm.weight"));
    assert!(remapped.contains_key("output.weight"));
    assert!(remapped.contains_key("layers.0.attention.q_proj.weight"));
    assert!(remapped.contains_key("layers.1.attention.out_proj.weight"));
    assert!(remapped.contains_key("layers.0.feed_forward.up_proj.weight"));
    assert!(remapped.contains_key("layers.1.ffn_norm.weight"));

    // HF names should not be present
    assert!(!remapped.contains_key("model.embed_tokens.weight"));
    assert!(!remapped.contains_key("lm_head.weight"));
}

#[test]
fn weight_map_validate_catches_missing() {
    let wmap = WeightMap::llama2(1);

    // Provide only global weights, missing per-layer
    let mut weights = HashMap::new();
    weights.insert("token_embedding.weight".to_string(), make_tensor(&[100, 32]));
    weights.insert("norm.weight".to_string(), make_tensor(&[32]));
    weights.insert("output.weight".to_string(), make_tensor(&[100, 32]));

    let result = wmap.validate(&weights, 1);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("attention.q_proj.weight"));
}

#[test]
fn weight_map_validate_passes_when_complete() {
    let n_layers = 1;
    let wmap = WeightMap::llama2(n_layers);

    let mut weights = HashMap::new();
    weights.insert("token_embedding.weight".to_string(), make_tensor(&[100, 32]));
    weights.insert("norm.weight".to_string(), make_tensor(&[32]));
    weights.insert("output.weight".to_string(), make_tensor(&[100, 32]));

    for i in 0..n_layers {
        weights.insert(format!("layers.{}.attention.q_proj.weight", i), make_tensor(&[32, 32]));
        weights.insert(format!("layers.{}.attention.k_proj.weight", i), make_tensor(&[32, 32]));
        weights.insert(format!("layers.{}.attention.v_proj.weight", i), make_tensor(&[32, 32]));
        weights.insert(format!("layers.{}.attention.out_proj.weight", i), make_tensor(&[32, 32]));
        weights.insert(format!("layers.{}.attention_norm.weight", i), make_tensor(&[32]));
        weights.insert(format!("layers.{}.ffn_norm.weight", i), make_tensor(&[32]));
        weights.insert(format!("layers.{}.feed_forward.up_proj.weight", i), make_tensor(&[64, 32]));
        weights.insert(format!("layers.{}.feed_forward.down_proj.weight", i), make_tensor(&[32, 64]));
    }

    assert!(wmap.validate(&weights, n_layers).is_ok());
}

#[test]
fn from_pretrained_produces_correct_output_shape() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 100,
        norm_eps: 1e-6,
        max_seq_len: 64,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };

    // Build internal-name weights for from_pretrained
    let mut weights = HashMap::new();
    weights.insert("token_embedding.weight".to_string(), make_tensor(&[100, 32]));
    weights.insert("norm.weight".to_string(), make_tensor(&[32]));
    weights.insert("output.weight".to_string(), make_tensor(&[100, 32]));

    weights.insert("layers.0.attention.q_proj.weight".to_string(), make_tensor(&[32, 32]));
    weights.insert("layers.0.attention.k_proj.weight".to_string(), make_tensor(&[32, 32]));
    weights.insert("layers.0.attention.v_proj.weight".to_string(), make_tensor(&[32, 32]));
    weights.insert("layers.0.attention.out_proj.weight".to_string(), make_tensor(&[32, 32]));
    weights.insert("layers.0.attention_norm.weight".to_string(), make_tensor(&[32]));
    weights.insert("layers.0.ffn_norm.weight".to_string(), make_tensor(&[32]));
    weights.insert("layers.0.feed_forward.up_proj.weight".to_string(), make_tensor(&[64, 32]));
    weights.insert("layers.0.feed_forward.down_proj.weight".to_string(), make_tensor(&[32, 64]));

    let model = LlmModel::from_pretrained(&config, weights).unwrap();

    // Forward pass with [2, 8] input
    let input_data: Vec<u8> = (0..2*8)
        .flat_map(|i| (i as f32 % 100.0).to_ne_bytes())
        .collect();
    let input = Tensor::new(input_data, vec![2, 8], DType::F32);

    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 8, 100]);
}

#[test]
fn from_pretrained_matches_new_output_shape() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 100,
        norm_eps: 1e-6,
        max_seq_len: 64,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };

    // Build from_pretrained model
    let mut weights = HashMap::new();
    weights.insert("token_embedding.weight".to_string(), make_tensor(&[100, 32]));
    weights.insert("norm.weight".to_string(), make_tensor(&[32]));
    weights.insert("output.weight".to_string(), make_tensor(&[100, 32]));
    weights.insert("layers.0.attention.q_proj.weight".to_string(), make_tensor(&[32, 32]));
    weights.insert("layers.0.attention.k_proj.weight".to_string(), make_tensor(&[32, 32]));
    weights.insert("layers.0.attention.v_proj.weight".to_string(), make_tensor(&[32, 32]));
    weights.insert("layers.0.attention.out_proj.weight".to_string(), make_tensor(&[32, 32]));
    weights.insert("layers.0.attention_norm.weight".to_string(), make_tensor(&[32]));
    weights.insert("layers.0.ffn_norm.weight".to_string(), make_tensor(&[32]));
    weights.insert("layers.0.feed_forward.up_proj.weight".to_string(), make_tensor(&[64, 32]));
    weights.insert("layers.0.feed_forward.down_proj.weight".to_string(), make_tensor(&[32, 64]));

    let pretrained_model = LlmModel::from_pretrained(&config, weights).unwrap();
    let new_model = LlmModel::new(&config).unwrap();

    // Both should produce the same output shape
    let input_data: Vec<u8> = (0..2*4)
        .flat_map(|i| (i as f32 % 100.0).to_ne_bytes())
        .collect();
    let input = Tensor::new(input_data, vec![2, 4], DType::F32);

    let out_pretrained = pretrained_model.forward(&input).unwrap();
    let out_new = new_model.forward(&input).unwrap();

    assert_eq!(out_pretrained.shape(), out_new.shape());
    assert_eq!(out_pretrained.shape(), &[2, 4, 100]);
}
