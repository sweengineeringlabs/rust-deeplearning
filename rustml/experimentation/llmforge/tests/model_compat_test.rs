use llmforge::config::{ModelConfig, PositionEncoding};
use llmforge::core::tensor::{Tensor, DType, f32_vec_to_bytes};
use llmforge::loader::{ModelLoader, WeightMap};
use llmforge::loader::gguf::{GGUFFile, GGUFValue, GGMLType, GGUF_MAGIC, gguf_weight_map};
use llmforge::models::LlmModel;
use llmforge::nn::{Embedding, Linear, LayerNorm, Layer};
use std::collections::HashMap;
use std::io::Write;
use tempfile::NamedTempFile;

fn make_f32_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    let bytes = f32_vec_to_bytes(data.to_vec());
    Tensor::new(bytes, shape, DType::F32)
}

// ===================== Step 3: Llama-2 E2E =====================

#[test]
fn hf_llama2_config_parsing() {
    let config_json = r#"{
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-06,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0
    }"#;

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(config_json.as_bytes()).unwrap();

    let config = ModelConfig::from_hf_llama2(file.path()).unwrap();
    assert_eq!(config.dim, 4096);
    assert_eq!(config.hidden_dim, 11008);
    assert_eq!(config.n_layers, 32);
    assert_eq!(config.n_heads, 32);
    assert_eq!(config.n_kv_heads, Some(32));
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.position_encoding, PositionEncoding::RoPE);
    assert_eq!(config.use_bias, Some(false));
}

#[test]
fn hf_llama2_config_rope_settings() {
    let config_json = r#"{
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "vocab_size": 100,
        "rms_norm_eps": 1e-05,
        "max_position_embeddings": 512,
        "rope_theta": 500000.0
    }"#;

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(config_json.as_bytes()).unwrap();

    let config = ModelConfig::from_hf_llama2(file.path()).unwrap();
    assert_eq!(config.position_encoding, PositionEncoding::RoPE);
    assert_eq!(config.rope_theta, 500000.0);
    assert!(config.causal);
}

#[test]
fn hf_llama2_gqa_config() {
    let config_json = r#"{
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 100,
        "rms_norm_eps": 1e-05,
        "max_position_embeddings": 512
    }"#;

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(config_json.as_bytes()).unwrap();

    let config = ModelConfig::from_hf_llama2(file.path()).unwrap();
    assert_eq!(config.n_kv_heads, Some(2));
    assert_eq!(config.rope_theta, 10000.0); // default when not specified
}

#[test]
fn llama2_forward_with_rope() {
    let config = ModelConfig {
        dim: 64,
        hidden_dim: 128,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: Some(4),
        vocab_size: 32,
        norm_eps: 1e-5,
        max_seq_len: 64,
        use_bias: Some(false),
        position_encoding: PositionEncoding::RoPE,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };

    let model = LlmModel::new(&config).unwrap();
    let input_data = vec![0.0f32, 1.0, 2.0];
    let input = make_f32_tensor(&input_data, vec![1, 3]);
    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 3, 32]);
}

#[test]
fn pos_embedding_loading_from_weights() {
    // Create config with Learned position encoding
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 16,
        norm_eps: 1e-5,
        max_seq_len: 8,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };

    let d = config.dim;
    let mut weights = HashMap::new();

    // Required weights for from_pretrained
    weights.insert("token_embedding.weight".to_string(), make_f32_tensor(&vec![0.1; 16 * d], vec![16, d]));
    weights.insert("output.weight".to_string(), make_f32_tensor(&vec![0.1; 16 * d], vec![16, d]));
    weights.insert("norm.weight".to_string(), make_f32_tensor(&vec![1.0; d], vec![d]));

    // Position embedding weight - should be loaded instead of random
    let pos_data: Vec<f32> = (0..8 * d).map(|i| i as f32 * 0.01).collect();
    weights.insert("pos_embedding.weight".to_string(), make_f32_tensor(&pos_data, vec![8, d]));

    // Layer weights
    let attn_d = d;
    weights.insert("layers.0.attention.q_proj.weight".to_string(), make_f32_tensor(&vec![0.01; attn_d * d], vec![attn_d, d]));
    weights.insert("layers.0.attention.k_proj.weight".to_string(), make_f32_tensor(&vec![0.01; attn_d * d], vec![attn_d, d]));
    weights.insert("layers.0.attention.v_proj.weight".to_string(), make_f32_tensor(&vec![0.01; attn_d * d], vec![attn_d, d]));
    weights.insert("layers.0.attention.out_proj.weight".to_string(), make_f32_tensor(&vec![0.01; d * d], vec![d, d]));
    weights.insert("layers.0.feed_forward.up_proj.weight".to_string(), make_f32_tensor(&vec![0.01; 64 * d], vec![64, d]));
    weights.insert("layers.0.feed_forward.down_proj.weight".to_string(), make_f32_tensor(&vec![0.01; d * 64], vec![d, 64]));
    weights.insert("layers.0.attention_norm.weight".to_string(), make_f32_tensor(&vec![1.0; d], vec![d]));
    weights.insert("layers.0.ffn_norm.weight".to_string(), make_f32_tensor(&vec![1.0; d], vec![d]));

    let model = LlmModel::from_pretrained(&config, weights).unwrap();
    assert!(model.pos_embedding.is_some());

    // Verify the loaded weight matches our data
    let loaded_pos = model.pos_embedding.as_ref().unwrap();
    let loaded_data = loaded_pos.weight.as_slice_f32().unwrap();
    assert!((loaded_data[0] - 0.0).abs() < 1e-4);
    assert!((loaded_data[1] - 0.01).abs() < 1e-4);
}

#[test]
fn llama2_gate_proj_mapping() {
    // Verify gate_proj is correctly handled in weight map
    let wm = WeightMap::llama2(2);
    assert_eq!(wm.len(), 3 + 9 * 2); // 3 global + 9 per layer * 2 layers
}

// ===================== Step 4: GPT-2 Loading =====================

#[test]
fn hf_gpt2_config_parsing() {
    let config_json = r#"{
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "vocab_size": 50257,
        "n_positions": 1024,
        "layer_norm_epsilon": 1e-05
    }"#;

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(config_json.as_bytes()).unwrap();

    let config = ModelConfig::from_hf_gpt2(file.path()).unwrap();
    assert_eq!(config.dim, 768);
    assert_eq!(config.hidden_dim, 4 * 768);
    assert_eq!(config.n_layers, 12);
    assert_eq!(config.n_heads, 12);
    assert_eq!(config.vocab_size, 50257);
    assert_eq!(config.max_seq_len, 1024);
    assert_eq!(config.norm_eps, 1e-5);
    assert_eq!(config.use_bias, Some(true));
    assert_eq!(config.position_encoding, PositionEncoding::Learned);
}

#[test]
fn hf_gpt2_config_custom_hidden_dim() {
    let config_json = r#"{
        "n_embd": 256,
        "n_inner": 512,
        "n_layer": 4,
        "n_head": 4,
        "vocab_size": 1000,
        "n_positions": 128
    }"#;

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(config_json.as_bytes()).unwrap();

    let config = ModelConfig::from_hf_gpt2(file.path()).unwrap();
    assert_eq!(config.hidden_dim, 512);
}

#[test]
fn gpt2_weight_map_count() {
    let wm = WeightMap::gpt2(12);
    // 4 global + 12 per layer * 12 layers = 4 + 144 = 148
    assert_eq!(wm.len(), 4 + 12 * 12);
}

#[test]
fn gpt2_qkv_split_shapes() {
    let d = 64;
    // Fused QKV weight: [3*d, d]
    let qkv_data: Vec<f32> = (0..3 * d * d).map(|i| i as f32 * 0.001).collect();
    let qkv = make_f32_tensor(&qkv_data, vec![3 * d, d]);

    let q = qkv.slice_rows(0, d).unwrap();
    let k = qkv.slice_rows(d, 2 * d).unwrap();
    let v = qkv.slice_rows(2 * d, 3 * d).unwrap();

    assert_eq!(q.shape(), &[d, d]);
    assert_eq!(k.shape(), &[d, d]);
    assert_eq!(v.shape(), &[d, d]);

    // Verify data correctness
    let q_data = q.as_slice_f32().unwrap();
    let k_data = k.as_slice_f32().unwrap();
    assert!((q_data[0] - 0.0).abs() < 1e-6); // First element of Q
    assert!((k_data[0] - (d * d) as f32 * 0.001).abs() < 1e-3); // First element of K
}

#[test]
fn gpt2_qkv_bias_split() {
    let d = 32;
    let bias_data: Vec<f32> = (0..3 * d).map(|i| i as f32).collect();
    let bias = make_f32_tensor(&bias_data, vec![3 * d]);

    let q_b = bias.as_slice_f32().unwrap()[..d].to_vec();
    let k_b = bias.as_slice_f32().unwrap()[d..2 * d].to_vec();
    let v_b = bias.as_slice_f32().unwrap()[2 * d..].to_vec();

    assert_eq!(q_b.len(), d);
    assert_eq!(k_b.len(), d);
    assert_eq!(v_b.len(), d);
    assert!((q_b[0] - 0.0).abs() < 1e-6);
    assert!((k_b[0] - d as f32).abs() < 1e-6);
    assert!((v_b[0] - (2 * d) as f32).abs() < 1e-6);
}

#[test]
fn gpt2_from_pretrained_forward() {
    let d = 32;
    let h = 64;
    let n_heads = 4;
    let vocab = 16;
    let max_seq = 8;

    let config = ModelConfig {
        dim: d,
        hidden_dim: h,
        n_layers: 1,
        n_heads,
        n_kv_heads: None,
        vocab_size: vocab,
        norm_eps: 1e-5,
        max_seq_len: max_seq,
        use_bias: Some(true),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };

    let mut weights = HashMap::new();

    // Token and positional embeddings
    weights.insert("token_embedding.weight".to_string(), make_f32_tensor(&vec![0.01; vocab * d], vec![vocab, d]));
    weights.insert("pos_embedding.weight".to_string(), make_f32_tensor(&vec![0.01; max_seq * d], vec![max_seq, d]));

    // Final norm
    weights.insert("norm.weight".to_string(), make_f32_tensor(&vec![1.0; d], vec![d]));
    weights.insert("norm.bias".to_string(), make_f32_tensor(&vec![0.0; d], vec![d]));

    // Layer 0
    // GPT-2 Conv1D format: weights stored as [In, Out]
    // Fused QKV: [d, 3*d] (Conv1D), transposed to [3*d, d] inside from_pretrained_gpt2
    weights.insert("layers.0.attention.c_attn.weight".to_string(), make_f32_tensor(&vec![0.01; d * 3 * d], vec![d, 3 * d]));
    weights.insert("layers.0.attention.c_attn.bias".to_string(), make_f32_tensor(&vec![0.0; 3 * d], vec![3 * d]));
    weights.insert("layers.0.attention.out_proj.weight".to_string(), make_f32_tensor(&vec![0.01; d * d], vec![d, d]));
    weights.insert("layers.0.attention.out_proj.bias".to_string(), make_f32_tensor(&vec![0.0; d], vec![d]));
    weights.insert("layers.0.feed_forward.up_proj.weight".to_string(), make_f32_tensor(&vec![0.01; d * h], vec![d, h]));
    weights.insert("layers.0.feed_forward.up_proj.bias".to_string(), make_f32_tensor(&vec![0.0; h], vec![h]));
    weights.insert("layers.0.feed_forward.down_proj.weight".to_string(), make_f32_tensor(&vec![0.01; h * d], vec![h, d]));
    weights.insert("layers.0.feed_forward.down_proj.bias".to_string(), make_f32_tensor(&vec![0.0; d], vec![d]));
    weights.insert("layers.0.attention_norm.weight".to_string(), make_f32_tensor(&vec![1.0; d], vec![d]));
    weights.insert("layers.0.attention_norm.bias".to_string(), make_f32_tensor(&vec![0.0; d], vec![d]));
    weights.insert("layers.0.ffn_norm.weight".to_string(), make_f32_tensor(&vec![1.0; d], vec![d]));
    weights.insert("layers.0.ffn_norm.bias".to_string(), make_f32_tensor(&vec![0.0; d], vec![d]));

    let model = LlmModel::from_pretrained_gpt2(&config, weights).unwrap();
    let input = make_f32_tensor(&[0.0, 1.0, 2.0], vec![1, 3]);
    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 3, vocab]);
}

#[test]
fn gpt2_tied_embeddings() {
    let d = 32;
    let vocab = 16;

    let config = ModelConfig {
        dim: d,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: vocab,
        norm_eps: 1e-5,
        max_seq_len: 8,
        use_bias: Some(true),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };

    let mut weights = HashMap::new();
    let emb_data: Vec<f32> = (0..vocab * d).map(|i| i as f32 * 0.001).collect();
    weights.insert("token_embedding.weight".to_string(), make_f32_tensor(&emb_data, vec![vocab, d]));
    weights.insert("pos_embedding.weight".to_string(), make_f32_tensor(&vec![0.01; 8 * d], vec![8, d]));
    weights.insert("norm.weight".to_string(), make_f32_tensor(&vec![1.0; d], vec![d]));
    weights.insert("norm.bias".to_string(), make_f32_tensor(&vec![0.0; d], vec![d]));
    weights.insert("layers.0.attention.c_attn.weight".to_string(), make_f32_tensor(&vec![0.01; d * 3 * d], vec![d, 3 * d]));
    weights.insert("layers.0.attention.c_attn.bias".to_string(), make_f32_tensor(&vec![0.0; 3 * d], vec![3 * d]));
    weights.insert("layers.0.attention.out_proj.weight".to_string(), make_f32_tensor(&vec![0.01; d * d], vec![d, d]));
    weights.insert("layers.0.attention.out_proj.bias".to_string(), make_f32_tensor(&vec![0.0; d], vec![d]));
    weights.insert("layers.0.feed_forward.up_proj.weight".to_string(), make_f32_tensor(&vec![0.01; d * 64], vec![d, 64]));
    weights.insert("layers.0.feed_forward.up_proj.bias".to_string(), make_f32_tensor(&vec![0.0; 64], vec![64]));
    weights.insert("layers.0.feed_forward.down_proj.weight".to_string(), make_f32_tensor(&vec![0.01; 64 * d], vec![64, d]));
    weights.insert("layers.0.feed_forward.down_proj.bias".to_string(), make_f32_tensor(&vec![0.0; d], vec![d]));
    weights.insert("layers.0.attention_norm.weight".to_string(), make_f32_tensor(&vec![1.0; d], vec![d]));
    weights.insert("layers.0.attention_norm.bias".to_string(), make_f32_tensor(&vec![0.0; d], vec![d]));
    weights.insert("layers.0.ffn_norm.weight".to_string(), make_f32_tensor(&vec![1.0; d], vec![d]));
    weights.insert("layers.0.ffn_norm.bias".to_string(), make_f32_tensor(&vec![0.0; d], vec![d]));

    let model = LlmModel::from_pretrained_gpt2(&config, weights).unwrap();

    // Tied: output weight should match token_embedding weight
    let tok_data = model.token_embedding.weight.as_slice_f32().unwrap();
    let out_data = model.output.weight.as_slice_f32().unwrap();
    assert_eq!(tok_data.len(), out_data.len());
    for (a, b) in tok_data.iter().zip(out_data.iter()) {
        assert!((a - b).abs() < 1e-6, "Tied embedding mismatch: {} vs {}", a, b);
    }
}

#[test]
fn gpt2_layernorm_with_bias() {
    let d = 32;
    let weight_data = vec![2.0f32; d];
    let bias_data = vec![0.5f32; d];
    let weight = make_f32_tensor(&weight_data, vec![d]);
    let bias = make_f32_tensor(&bias_data, vec![d]);

    let ln = LayerNorm::from_weights(weight, bias, 1e-5);
    assert_eq!(ln.weight.shape(), &[d]);
    assert_eq!(ln.bias.shape(), &[d]);

    let input = make_f32_tensor(&vec![1.0; d], vec![1, d]);
    let output = ln.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, d]);
}

// ===================== Step 5: GGUF Support =====================

/// Build a minimal valid GGUF file in memory for testing.
fn build_gguf_bytes(
    version: u32,
    metadata: &[(&str, u32, &[u8])], // (key, type_id, value_bytes)
    tensor_infos: &[(&str, &[usize], u32, u64)], // (name, dims, ggml_type, offset)
    tensor_data: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(&GGUF_MAGIC);
    // Version
    buf.extend_from_slice(&version.to_le_bytes());
    // Tensor count
    buf.extend_from_slice(&(tensor_infos.len() as u64).to_le_bytes());
    // Metadata count
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    // Metadata KV pairs
    for &(key, type_id, value_bytes) in metadata {
        // Key string: len (u64) + bytes
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        // Type ID
        buf.extend_from_slice(&type_id.to_le_bytes());
        // Value bytes (pre-encoded)
        buf.extend_from_slice(value_bytes);
    }

    // Tensor info entries
    for &(name, dims, ggml_type, offset) in tensor_infos {
        // Name string
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        // n_dims
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        // Dimensions
        for &d in dims {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        // GGML type
        buf.extend_from_slice(&ggml_type.to_le_bytes());
        // Offset
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    // Pad to 32-byte alignment
    let data_offset = (buf.len() + 31) & !31;
    while buf.len() < data_offset {
        buf.push(0);
    }

    // Tensor data
    buf.extend_from_slice(tensor_data);

    buf
}

#[test]
fn gguf_magic_validation() {
    let bad_data = b"GGML\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    let result = GGUFFile::parse_bytes(bad_data);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("magic"), "Error should mention magic: {}", err_msg);
}

#[test]
fn gguf_unsupported_version() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC);
    data.extend_from_slice(&1u32.to_le_bytes()); // version 1 not supported
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata count
    let result = GGUFFile::parse_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn gguf_empty_file_valid() {
    let data = build_gguf_bytes(2, &[], &[], &[]);
    let gguf = GGUFFile::parse_bytes(&data).unwrap();
    assert_eq!(gguf.version, 2);
    assert_eq!(gguf.metadata.len(), 0);
    assert_eq!(gguf.tensor_infos.len(), 0);
}

#[test]
fn gguf_version_3_valid() {
    let data = build_gguf_bytes(3, &[], &[], &[]);
    let gguf = GGUFFile::parse_bytes(&data).unwrap();
    assert_eq!(gguf.version, 3);
}

#[test]
fn gguf_metadata_u32() {
    let val_bytes = 42u32.to_le_bytes();
    let data = build_gguf_bytes(2, &[("test.key", 4, &val_bytes)], &[], &[]);
    let gguf = GGUFFile::parse_bytes(&data).unwrap();

    assert_eq!(gguf.metadata.len(), 1);
    let val = gguf.metadata.get("test.key").unwrap();
    assert_eq!(val.as_u32(), Some(42));
}

#[test]
fn gguf_metadata_f32() {
    let val_bytes = 3.14f32.to_le_bytes();
    let data = build_gguf_bytes(2, &[("test.float", 6, &val_bytes)], &[], &[]);
    let gguf = GGUFFile::parse_bytes(&data).unwrap();

    let val = gguf.metadata.get("test.float").unwrap();
    assert!((val.as_f32().unwrap() - 3.14).abs() < 0.001);
}

#[test]
fn gguf_metadata_string() {
    let s = "hello world";
    let mut val_bytes = Vec::new();
    val_bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
    val_bytes.extend_from_slice(s.as_bytes());

    let data = build_gguf_bytes(2, &[("test.str", 8, &val_bytes)], &[], &[]);
    let gguf = GGUFFile::parse_bytes(&data).unwrap();

    let val = gguf.metadata.get("test.str").unwrap();
    assert_eq!(val.as_string(), Some("hello world"));
}

#[test]
fn gguf_tensor_info_parsing() {
    // One F32 tensor: shape [4, 8], 128 bytes
    let tensor_data = vec![0u8; 128]; // 4*8*4 = 128 bytes

    let data = build_gguf_bytes(
        2,
        &[],
        &[("test_tensor", &[8, 4], 0, 0)], // F32 type = 0
        &tensor_data,
    );
    let gguf = GGUFFile::parse_bytes(&data).unwrap();

    assert_eq!(gguf.tensor_infos.len(), 1);
    let info = &gguf.tensor_infos[0];
    assert_eq!(info.name, "test_tensor");
    assert_eq!(info.dimensions, vec![8, 4]);
    assert_eq!(info.ggml_type, GGMLType::F32);
    assert_eq!(info.offset, 0);
}

#[test]
fn gguf_config_extraction() {
    // Build metadata for a minimal Llama config
    let mut meta_entries = Vec::new();

    let dim_bytes = 128u32.to_le_bytes();
    meta_entries.push(("llama.embedding_length", 4u32, dim_bytes.as_slice()));

    let heads_bytes = 4u32.to_le_bytes();
    meta_entries.push(("llama.attention.head_count", 4u32, heads_bytes.as_slice()));

    let layers_bytes = 2u32.to_le_bytes();
    meta_entries.push(("llama.block_count", 4u32, layers_bytes.as_slice()));

    let data = build_gguf_bytes(2, &meta_entries, &[], &[]);
    let gguf = GGUFFile::parse_bytes(&data).unwrap();
    let config = gguf.to_model_config().unwrap();

    assert_eq!(config.dim, 128);
    assert_eq!(config.n_heads, 4);
    assert_eq!(config.n_layers, 2);
    assert_eq!(config.position_encoding, PositionEncoding::RoPE);
}

#[test]
fn gguf_weight_name_mapping() {
    let wm = gguf_weight_map(2);
    let mut gguf_weights = HashMap::new();
    gguf_weights.insert("token_embd.weight".to_string(), make_f32_tensor(&[1.0; 4], vec![2, 2]));
    gguf_weights.insert("output_norm.weight".to_string(), make_f32_tensor(&[1.0; 2], vec![2]));
    gguf_weights.insert("blk.0.attn_q.weight".to_string(), make_f32_tensor(&[1.0; 4], vec![2, 2]));

    let remapped = wm.remap(gguf_weights).unwrap();
    assert!(remapped.contains_key("token_embedding.weight"));
    assert!(remapped.contains_key("norm.weight"));
    assert!(remapped.contains_key("layers.0.attention.q_proj.weight"));
}

#[test]
fn gguf_f32_tensor_loading() {
    // Create a GGUF file with one F32 tensor [2, 4]
    let tensor_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let data = build_gguf_bytes(
        2,
        &[],
        &[("weights", &[4, 2], 0, 0)], // F32, dims stored as [inner, outer]
        &tensor_data,
    );

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&data).unwrap();

    let gguf = GGUFFile::parse_bytes(&data).unwrap();
    let tensors = gguf.load_tensors(file.path()).unwrap();

    let t = tensors.get("weights").unwrap();
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.shape(), &[2, 4]); // reversed from GGUF order
    let vals = t.as_slice_f32().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!((vals[7] - 8.0).abs() < 1e-6);
}

#[test]
fn gguf_q8_0_tensor_loading() {
    // One Q8_0 tensor: 32 elements = 1 block = 34 bytes
    let mut block_data = vec![0u8; 34];
    // Scale (f16): 1.0
    let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
    block_data[0] = scale_bytes[0];
    block_data[1] = scale_bytes[1];
    // i8 values: 0..31
    for i in 0..32 {
        block_data[2 + i] = i as u8;
    }

    let data = build_gguf_bytes(
        2,
        &[],
        &[("q8_weights", &[32], 8, 0)], // Q8_0 = type 8
        &block_data,
    );

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&data).unwrap();

    let gguf = GGUFFile::parse_bytes(&data).unwrap();
    let tensors = gguf.load_tensors(file.path()).unwrap();

    let t = tensors.get("q8_weights").unwrap();
    assert_eq!(t.dtype(), DType::Q8_0);
    assert_eq!(t.shape(), &[32]);
}

#[test]
fn gguf_q4_0_tensor_loading() {
    // One Q4_0 tensor: 32 elements = 1 block = 18 bytes
    let mut block_data = vec![0u8; 18];
    // Scale (f16): 0.5
    let scale_bytes = half::f16::from_f32(0.5).to_le_bytes();
    block_data[0] = scale_bytes[0];
    block_data[1] = scale_bytes[1];
    // Packed 4-bit values
    for i in 0..16 {
        block_data[2 + i] = 0x88; // both nibbles = 8, which is 0 after -8 offset
    }

    let data = build_gguf_bytes(
        2,
        &[],
        &[("q4_weights", &[32], 2, 0)], // Q4_0 = type 2
        &block_data,
    );

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&data).unwrap();

    let gguf = GGUFFile::parse_bytes(&data).unwrap();
    let tensors = gguf.load_tensors(file.path()).unwrap();

    let t = tensors.get("q4_weights").unwrap();
    assert_eq!(t.dtype(), DType::Q4_0);
    assert_eq!(t.shape(), &[32]);
}

#[test]
fn gguf_unsupported_type_errors() {
    // Q4_1 = type 3, not supported
    let data = build_gguf_bytes(
        2,
        &[],
        &[("bad_tensor", &[32], 3, 0)], // Q4_1 = unsupported for loading
        &vec![0u8; 20],
    );

    let gguf = GGUFFile::parse_bytes(&data).unwrap();
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&data).unwrap();
    let result = gguf.load_tensors(file.path());
    assert!(result.is_err());
}

#[test]
fn gguf_multiple_tensors() {
    // Two F32 tensors: [4] (16 bytes) at offset 0, [2] (8 bytes) at offset 16
    let mut tensor_data = Vec::new();
    for v in &[1.0f32, 2.0, 3.0, 4.0] {
        tensor_data.extend_from_slice(&v.to_le_bytes());
    }
    for v in &[5.0f32, 6.0] {
        tensor_data.extend_from_slice(&v.to_le_bytes());
    }

    let data = build_gguf_bytes(
        2,
        &[],
        &[
            ("tensor_a", &[4], 0, 0),   // F32 at offset 0
            ("tensor_b", &[2], 0, 16),  // F32 at offset 16
        ],
        &tensor_data,
    );

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&data).unwrap();

    let gguf = GGUFFile::parse_bytes(&data).unwrap();
    let tensors = gguf.load_tensors(file.path()).unwrap();

    assert_eq!(tensors.len(), 2);
    let a = tensors.get("tensor_a").unwrap();
    let b = tensors.get("tensor_b").unwrap();
    assert_eq!(a.as_slice_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(b.as_slice_f32().unwrap(), &[5.0, 6.0]);
}

#[test]
fn gguf_header_from_file() {
    let data = build_gguf_bytes(2, &[], &[], &[]);
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&data).unwrap();

    let gguf = GGUFFile::parse_header(file.path()).unwrap();
    assert_eq!(gguf.version, 2);
}

#[test]
fn slice_rows_basic() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let tensor = make_f32_tensor(&data, vec![3, 4]);

    let slice = tensor.slice_rows(1, 3).unwrap();
    assert_eq!(slice.shape(), &[2, 4]);
    let vals = slice.as_slice_f32().unwrap();
    assert_eq!(vals, &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn slice_rows_out_of_bounds() {
    let data = vec![1.0f32; 12];
    let tensor = make_f32_tensor(&data, vec![3, 4]);
    assert!(tensor.slice_rows(2, 5).is_err());
}

#[test]
fn slice_rows_requires_2d() {
    let data = vec![1.0f32; 8];
    let tensor = make_f32_tensor(&data, vec![2, 2, 2]);
    assert!(tensor.slice_rows(0, 1).is_err());
}
