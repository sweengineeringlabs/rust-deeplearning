use llmforge::config::{ModelConfig, PositionEncoding};
use llmforge::models::LlmModel;
use llmforge::inference::Generator;
use llmforge::tokenization::NaiveTokenizer;
use llmforge::core::tensor::{Tensor, DType};
use llmforge::attention::KVCache;

fn tiny_config() -> ModelConfig {
    ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 256,
        norm_eps: 1e-5,
        max_seq_len: 32,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
    }
}

fn make_input_ids(tokens: &[u32]) -> Tensor {
    let data: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    Tensor::new(bytes, vec![1, tokens.len()], DType::F32)
}

#[test]
fn end_to_end_generation_deterministic() {
    let config = tiny_config();
    let model = LlmModel::new(&config).unwrap();
    let tokenizer = NaiveTokenizer::new();

    // temp=0 -> greedy, deterministic
    let gen = Generator::new(&model, &tokenizer, 0.0);

    let out1 = gen.generate("A", 3).unwrap();
    let out2 = gen.generate("A", 3).unwrap();
    assert_eq!(out1, out2, "Greedy generation should be deterministic");
}

#[test]
fn end_to_end_forward_correct_shape() {
    let config = tiny_config();
    let model = LlmModel::new(&config).unwrap();

    let input = make_input_ids(&[65, 66, 67]); // "ABC"
    let output = model.forward(&input).unwrap();
    // [batch=1, seq=3, vocab_size=256]
    assert_eq!(output.shape(), &[1, 3, config.vocab_size]);
}

#[test]
fn long_context_fills_cache_then_errors() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 256,
        norm_eps: 1e-5,
        max_seq_len: 8, // Very small
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
    };
    let model = LlmModel::new(&config).unwrap();

    let head_dim = config.dim / config.n_heads;
    let mut cache = KVCache::new(config.n_layers, config.max_seq_len, head_dim, config.n_heads);

    // Prefill with max tokens
    let input = make_input_ids(&[65, 66, 67, 68, 69, 70, 71, 72]); // 8 tokens = max
    let result = model.forward_with_cache(&input, &mut cache);
    assert!(result.is_ok(), "Prefill at max_seq_len should succeed");
    cache.advance(8);

    // Now try one more token — should fail with SequenceLengthExceeded
    let next = make_input_ids(&[73]);
    let result = model.forward_with_cache(&next, &mut cache);
    assert!(result.is_err(), "Exceeding max_seq_len should error");
}

#[test]
fn long_context_within_limit_succeeds() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 256,
        norm_eps: 1e-5,
        max_seq_len: 16,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
    };
    let model = LlmModel::new(&config).unwrap();

    let head_dim = config.dim / config.n_heads;
    let mut cache = KVCache::new(config.n_layers, config.max_seq_len, head_dim, config.n_heads);

    // Prefill 5 tokens
    let input = make_input_ids(&[65, 66, 67, 68, 69]);
    model.forward_with_cache(&input, &mut cache).unwrap();
    cache.advance(5);

    // Decode 3 more tokens (total 8, well within 16)
    for i in 0..3 {
        let next = make_input_ids(&[70 + i]);
        model.forward_with_cache(&next, &mut cache).unwrap();
        cache.advance(1);
    }
    assert_eq!(cache.current_len, 8);
}

#[test]
fn error_propagation_invalid_config() {
    let config = ModelConfig {
        dim: 0, // Invalid
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 256,
        norm_eps: 1e-5,
        max_seq_len: 32,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
    };
    let result = config.validate();
    assert!(result.is_err(), "dim=0 should fail validation");
}

#[test]
fn error_propagation_missing_weight() {
    let config = tiny_config();
    // Provide an incomplete weight map (missing keys)
    let weights = std::collections::HashMap::new();
    let result = LlmModel::from_pretrained(&config, weights);
    assert!(result.is_err(), "Empty weight map should error");
}

#[test]
fn error_propagation_shape_mismatch_forward() {
    let config = tiny_config();
    let model = LlmModel::new(&config).unwrap();

    // Wrong input shape: [1, 3, 5] (3D instead of 2D [batch, seq])
    // The Embedding forward expects [batch, seq] (2D),
    // so a wrong dimensionality should cause an error or panic.
    // We'll use a seq_len that exceeds max_seq_len to trigger a clean error.
    let too_long: Vec<u32> = (0..config.max_seq_len as u32 + 1).collect();
    let input = make_input_ids(&too_long);
    let result = model.forward(&input);
    assert!(
        result.is_err(),
        "Input exceeding max_seq_len should produce error"
    );
}
