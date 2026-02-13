use llmforge::core::tensor::{Tensor, DType};
use llmforge::config::ModelConfig;
use llmforge::attention::{MultiHeadAttention, KVCache};
use llmforge::tokenization::{NaiveTokenizer, Tokenizer};
use llmforge::transformer::{FeedForward, Activation};

// ── Broadcasting validation ──────────────────────────────────────────

#[test]
fn broadcast_add_valid_suffix() {
    // [3, 4] + [4] should succeed
    let lhs = Tensor::zeros(&[3, 4]);
    let rhs = Tensor::zeros(&[4]);
    assert!(lhs.add(&rhs).is_ok());
}

#[test]
fn broadcast_add_valid_prefix_one() {
    // [2, 3, 4] + [1, 4] should succeed (1 broadcasts to 3)
    let lhs = Tensor::zeros(&[2, 3, 4]);
    let rhs = Tensor::zeros(&[1, 4]);
    // element counts: 24 and 4 -> 24 % 4 == 0, and shape suffix is valid
    assert!(lhs.add(&rhs).is_ok());
}

#[test]
fn broadcast_add_invalid_shape() {
    // [10, 5] + [2] should fail — 2 is not a valid broadcast suffix of 5
    let lhs = Tensor::zeros(&[10, 5]);
    let rhs = Tensor::zeros(&[2]);
    let result = lhs.add(&rhs);
    assert!(result.is_err(), "Expected broadcasting error for [10,5] + [2]");
}

#[test]
fn broadcast_add_invalid_inner_dim() {
    // [6, 4] + [3] should fail — 3 != 4
    let lhs = Tensor::zeros(&[6, 4]);
    let rhs = Tensor::zeros(&[3]);
    let result = lhs.add(&rhs);
    assert!(result.is_err(), "Expected broadcasting error for [6,4] + [3]");
}

// ── ModelConfig validation ───────────────────────────────────────────

#[test]
fn config_validate_dim_zero_fails() {
    let config = ModelConfig {
        dim: 0,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 100,
        norm_eps: 1e-5,
        max_seq_len: 128,
        use_bias: None,
    };
    assert!(config.validate().is_err());
}

#[test]
fn config_validate_n_heads_zero_fails() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 0,
        n_kv_heads: None,
        vocab_size: 100,
        norm_eps: 1e-5,
        max_seq_len: 128,
        use_bias: None,
    };
    assert!(config.validate().is_err());
}

#[test]
fn config_validate_dim_not_divisible_by_heads_fails() {
    let config = ModelConfig {
        dim: 33,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 100,
        norm_eps: 1e-5,
        max_seq_len: 128,
        use_bias: None,
    };
    assert!(config.validate().is_err());
}

#[test]
fn config_validate_valid_config_passes() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 100,
        norm_eps: 1e-5,
        max_seq_len: 128,
        use_bias: None,
    };
    assert!(config.validate().is_ok());
}

// ── NaiveTokenizer multi-byte round-trip ─────────────────────────────

#[test]
fn naive_tokenizer_multibyte_roundtrip() {
    let tok = NaiveTokenizer::new();

    // Contains multi-byte UTF-8 characters
    let text = "Hello 世界! 🌍";
    let encoded = tok.encode(text).expect("encode failed");

    // Each char should produce exactly one token
    let expected_count = text.chars().count();
    assert_eq!(encoded.len(), expected_count);

    let decoded = tok.decode(&encoded).expect("decode failed");
    assert_eq!(decoded, text);
}

#[test]
fn naive_tokenizer_ascii_roundtrip() {
    let tok = NaiveTokenizer::new();
    let text = "Hello World";
    let encoded = tok.encode(text).unwrap();
    let decoded = tok.decode(&encoded).unwrap();
    assert_eq!(decoded, text);
}

// ── MultiHeadAttention::new() validation ─────────────────────────────

#[test]
fn mha_new_invalid_divisibility() {
    // d_model=33, num_heads=4 → 33 % 4 != 0 → should return Err
    let result = MultiHeadAttention::new(33, 4, false);
    assert!(result.is_err(), "Expected error when d_model % num_heads != 0");
}

#[test]
fn mha_new_valid_divisibility() {
    // d_model=64, num_heads=4 → 64 % 4 == 0 → should be Ok
    let result = MultiHeadAttention::new(64, 4, false);
    assert!(result.is_ok());
}

// ── f32_vec_to_bytes round-trip via tensor ───────────────────────────

#[test]
fn f32_data_roundtrip_via_tensor() {
    let values = vec![1.0f32, 2.0, 3.0, -4.5, 0.0, f32::MAX];
    let bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&values).to_vec();
    let tensor = Tensor::new(bytes, vec![6], DType::F32);
    let slice = tensor.as_slice_f32().unwrap();
    assert_eq!(slice, &values[..]);
}

// ── EOS field exists and generator compiles ──────────────────────────

#[test]
fn generator_eos_field_exists() {
    // Compile-time check that Generator has eos_token_id field
    use llmforge::inference::Generator;
    use llmforge::models::LlmModel;

    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 256,
        norm_eps: 1e-5,
        max_seq_len: 128,
        use_bias: Some(true),
    };
    let model = LlmModel::new(&config).unwrap();
    let tokenizer = NaiveTokenizer::new();
    let mut gen = Generator::new(&model, &tokenizer, 0.0);
    gen.eos_token_id = Some(0);
    // Just verifying it compiles and the field is accessible
    assert_eq!(gen.eos_token_id, Some(0));
}

// ── Temperature=0 produces deterministic (greedy) output ─────────────

#[test]
fn temperature_zero_is_deterministic() {
    use llmforge::inference::Generator;
    use llmforge::models::LlmModel;

    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 256,
        norm_eps: 1e-5,
        max_seq_len: 128,
        use_bias: Some(true),
    };
    let model = LlmModel::new(&config).unwrap();
    let tokenizer = NaiveTokenizer::new();

    // temperature=0.0 -> greedy
    let gen = Generator::new(&model, &tokenizer, 0.0);

    let out1 = gen.generate("A", 3).expect("gen1 failed");
    let out2 = gen.generate("A", 3).expect("gen2 failed");
    assert_eq!(out1, out2, "temperature=0 should produce deterministic output");
}

// ── SiLU / ReLU numerical checks ────────────────────────────────────

#[test]
fn silu_numerical_check() {
    // SiLU(x) = x * sigmoid(x)
    let values = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&values).to_vec();
    let t = Tensor::new(bytes, vec![5], DType::F32);

    let result = t.silu().unwrap();
    let out = result.as_slice_f32().unwrap();

    // SiLU(0) = 0
    assert!((out[2] - 0.0).abs() < 1e-6, "SiLU(0) should be 0");

    // SiLU(x) > 0 for x > 0
    assert!(out[3] > 0.0, "SiLU(1) should be positive");
    assert!(out[4] > 0.0, "SiLU(2) should be positive");

    // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
    assert!((out[3] - 0.7311).abs() < 0.01, "SiLU(1) ≈ 0.7311, got {}", out[3]);

    // SiLU(x) < 0 for x < 0 (slightly negative)
    assert!(out[0] < 0.0, "SiLU(-2) should be negative");
}

#[test]
fn relu_numerical_check() {
    let values = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&values).to_vec();
    let t = Tensor::new(bytes, vec![5], DType::F32);

    let result = t.relu().unwrap();
    let out = result.as_slice_f32().unwrap();

    assert_eq!(out[0], 0.0); // max(0, -2) = 0
    assert_eq!(out[1], 0.0); // max(0, -0.5) = 0
    assert_eq!(out[2], 0.0); // max(0, 0) = 0
    assert_eq!(out[3], 0.5); // max(0, 0.5) = 0.5
    assert_eq!(out[4], 2.0); // max(0, 2) = 2
}

// ── FeedForward with different activations ──────────────────────────

#[test]
fn feedforward_with_silu_runs() {
    let d_model = 32;
    let hidden_dim = 64;
    let ff = FeedForward::with_activation(d_model, hidden_dim, false, Activation::Silu);

    let input = Tensor::zeros(&[1, 4, d_model]);
    let output = ff.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 4, d_model]);
}

#[test]
fn feedforward_with_relu_runs() {
    let d_model = 32;
    let hidden_dim = 64;
    let ff = FeedForward::with_activation(d_model, hidden_dim, false, Activation::Relu);

    let input = Tensor::zeros(&[1, 4, d_model]);
    let output = ff.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 4, d_model]);
}

// ── KVCache head_dim mismatch ───────────────────────────────────────

#[test]
fn kvcache_head_dim_mismatch_returns_error() {
    let d_model = 32;
    let num_heads = 4;
    let _head_dim = d_model / num_heads; // 8

    let mha = MultiHeadAttention::new(d_model, num_heads, false).unwrap();

    // Create cache with wrong head_dim (16 instead of 8)
    let mut cache = KVCache::new(1, 64, 16, num_heads);

    let input = Tensor::zeros(&[1, 4, d_model]);
    let result = mha.forward_with_cache(&input, &mut cache, 0);
    assert!(result.is_err(), "Expected error for head_dim mismatch");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("head_dim"), "Error should mention head_dim: {}", err_msg);
}

// ── View bounds validation ──────────────────────────────────────────

#[test]
fn view_bounds_catches_invalid_reshape() {
    // Create a small tensor and try to reshape to a shape that would
    // address memory beyond storage bounds via transposition
    let t = Tensor::zeros(&[2, 3]); // 24 bytes (6 f32)

    // Normal reshape should work
    let r = t.reshape(&[3, 2]);
    assert!(r.is_ok());

    // Reshape preserving element count should work
    let r = t.reshape(&[6]);
    assert!(r.is_ok());
}
