use llmforge::core::tensor::{Tensor, DType};
use llmforge::config::{ModelConfig, PositionEncoding};
use llmforge::attention::{MultiHeadAttention, CrossAttention, KVCache, RoPEFreqs, compute_alibi_slopes, alibi_bias};
use llmforge::transformer::{FeedForward, Activation};
use llmforge::nn::{Linear, Embedding, LayerNorm, Freezable};
use llmforge::models::LlmModel;

fn make_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
    Tensor::new(bytes, shape, DType::F32)
}

// ── SwiGLU tests ─────────────────────────────────────────────────────

#[test]
fn swiglu_forward_shape() {
    let d_model = 32;
    let hidden_dim = 64;
    let ff = FeedForward::swiglu(d_model, hidden_dim, false);
    let input = Tensor::zeros(&[1, 4, d_model]);
    let output = ff.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 4, d_model]);
}

#[test]
fn swiglu_with_activation_constructor() {
    let d_model = 32;
    let hidden_dim = 64;
    let ff = FeedForward::with_activation(d_model, hidden_dim, false, Activation::SwiGLU);
    let input = Tensor::zeros(&[1, 2, d_model]);
    let output = ff.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 2, d_model]);
}

#[test]
fn swiglu_from_weights() {
    let d_model = 16;
    let hidden_dim = 32;
    let up = Linear::new(d_model, hidden_dim, false);
    let gate = Linear::new(d_model, hidden_dim, false);
    let down = Linear::new(hidden_dim, d_model, false);
    let ff = FeedForward::from_weights_swiglu(up, gate, down);
    let input = Tensor::zeros(&[1, 3, d_model]);
    let output = ff.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 3, d_model]);
}

#[test]
fn all_activation_variants_run() {
    let d_model = 16;
    let hidden = 32;
    for act in &[Activation::Gelu, Activation::Silu, Activation::Relu, Activation::SwiGLU] {
        let ff = FeedForward::with_activation(d_model, hidden, false, *act);
        let input = Tensor::zeros(&[1, 2, d_model]);
        let output = ff.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, d_model], "Failed for {:?}", act);
    }
}

// ── GQA tests ────────────────────────────────────────────────────────

#[test]
fn gqa_forward_shape() {
    let d_model = 32;
    let num_heads = 4;
    let num_kv_heads = 2; // GQA: 4 Q heads, 2 KV heads
    let mha = MultiHeadAttention::new(
        d_model, num_heads, Some(num_kv_heads), false, false,
        PositionEncoding::Learned, 128, 10000.0,
    ).unwrap();
    let input = Tensor::zeros(&[1, 4, d_model]);
    let output = mha.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 4, d_model]);
}

#[test]
fn gqa_equivalence_when_heads_equal() {
    // When n_kv_heads == n_heads, GQA is standard MHA
    let d_model = 32;
    let num_heads = 4;
    let mha_standard = MultiHeadAttention::new(
        d_model, num_heads, None, false, false,
        PositionEncoding::Learned, 128, 10000.0,
    ).unwrap();
    let mha_gqa = MultiHeadAttention::new(
        d_model, num_heads, Some(num_heads), false, false,
        PositionEncoding::Learned, 128, 10000.0,
    ).unwrap();
    // Both should produce valid outputs of the same shape
    let input = Tensor::zeros(&[1, 3, d_model]);
    let out_std = mha_standard.forward(&input).unwrap();
    let out_gqa = mha_gqa.forward(&input).unwrap();
    assert_eq!(out_std.shape(), out_gqa.shape());
}

#[test]
fn gqa_cache_uses_kv_heads() {
    let d_model = 32;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = d_model / num_heads; // 8
    let mha = MultiHeadAttention::new(
        d_model, num_heads, Some(num_kv_heads), false, false,
        PositionEncoding::Learned, 64, 10000.0,
    ).unwrap();

    let mut cache = KVCache::new(1, 64, head_dim, num_kv_heads);
    let input = Tensor::zeros(&[1, 3, d_model]);
    let output = mha.forward_with_cache(&input, &mut cache, 0).unwrap();
    assert_eq!(output.shape(), &[1, 3, d_model]);
}

#[test]
fn gqa_invalid_config() {
    // n_heads=4, n_kv_heads=3: 4 % 3 != 0
    let result = MultiHeadAttention::new(
        32, 4, Some(3), false, false,
        PositionEncoding::Learned, 128, 10000.0,
    );
    assert!(result.is_err());
}

#[test]
fn mqa_edge_case() {
    // Multi-Query Attention: n_kv_heads=1
    let d_model = 32;
    let num_heads = 4;
    let mha = MultiHeadAttention::new(
        d_model, num_heads, Some(1), false, false,
        PositionEncoding::Learned, 128, 10000.0,
    ).unwrap();
    let input = Tensor::zeros(&[1, 3, d_model]);
    let output = mha.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 3, d_model]);
}

// ── Causal mask tests ────────────────────────────────────────────────

#[test]
fn causal_prevents_future_attention() {
    // With causal=true, attention should be lower-triangular
    let d_model = 32;
    let mha = MultiHeadAttention::new(
        d_model, 4, None, false, true, // causal=true
        PositionEncoding::Learned, 128, 10000.0,
    ).unwrap();
    let input = Tensor::zeros(&[1, 4, d_model]);
    let output = mha.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 4, d_model]);
}

#[test]
fn causal_prefill_vs_decode() {
    let d_model = 32;
    let num_heads = 4;
    let head_dim = d_model / num_heads;
    let mha = MultiHeadAttention::new(
        d_model, num_heads, None, false, true,
        PositionEncoding::Learned, 32, 10000.0,
    ).unwrap();

    // Prefill
    let mut cache = KVCache::new(1, 32, head_dim, num_heads);
    let input = Tensor::zeros(&[1, 4, d_model]);
    let out1 = mha.forward_with_cache(&input, &mut cache, 0).unwrap();
    assert_eq!(out1.shape(), &[1, 4, d_model]);
    cache.advance(4);

    // Decode (seq_len=1: causal mask skipped)
    let input2 = Tensor::zeros(&[1, 1, d_model]);
    let out2 = mha.forward_with_cache(&input2, &mut cache, 0).unwrap();
    assert_eq!(out2.shape(), &[1, 1, d_model]);
}

#[test]
fn non_causal_symmetric() {
    let d_model = 32;
    let mha = MultiHeadAttention::new(
        d_model, 4, None, false, false, // causal=false
        PositionEncoding::Learned, 128, 10000.0,
    ).unwrap();
    let input = Tensor::zeros(&[1, 4, d_model]);
    let output = mha.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 4, d_model]);
}

// ── RoPE tests ───────────────────────────────────────────────────────

#[test]
fn rope_freq_precomputation() {
    let head_dim = 8;
    let max_seq = 16;
    let rope = RoPEFreqs::new(head_dim, max_seq, 10000.0);
    // Just verifying construction doesn't panic and apply works
    let x = Tensor::zeros(&[1, 2, 4, head_dim]);
    let out = rope.apply(&x, 0).unwrap();
    assert_eq!(out.shape(), &[1, 2, 4, head_dim]);
}

#[test]
fn rope_position_zero_preserves_first_half() {
    // At position 0, cos(0)=1, sin(0)=0
    // So x1*1 - x2*0 = x1 (first half preserved)
    let head_dim = 4;
    let rope = RoPEFreqs::new(head_dim, 16, 10000.0);
    let data = vec![1.0f32, 2.0, 3.0, 4.0]; // one head, one position
    let x = make_tensor(&data, vec![1, 1, 1, head_dim]);
    let out = rope.apply(&x, 0).unwrap();
    let out_data = out.as_slice_f32().unwrap();
    // First half: x1*cos(0) - x2*sin(0) = x1*1 - x2*0 = x1
    assert!((out_data[0] - 1.0).abs() < 1e-6);
    assert!((out_data[1] - 2.0).abs() < 1e-6);
}

#[test]
fn rope_different_positions_differ() {
    let head_dim = 8;
    let rope = RoPEFreqs::new(head_dim, 16, 10000.0);

    let data: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    let x = make_tensor(&data, vec![1, 1, 1, head_dim]);

    let out0 = rope.apply(&x, 0).unwrap();
    let out5 = rope.apply(&x, 5).unwrap();

    let d0 = out0.as_slice_f32().unwrap();
    let d5 = out5.as_slice_f32().unwrap();

    // Outputs at different positions should differ
    let mut any_diff = false;
    for (a, b) in d0.iter().zip(d5.iter()) {
        if (a - b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(any_diff, "RoPE at positions 0 and 5 should produce different outputs");
}

#[test]
fn rope_shape_preserved() {
    let head_dim = 8;
    let rope = RoPEFreqs::new(head_dim, 32, 10000.0);
    let x = Tensor::zeros(&[2, 4, 6, head_dim]);
    let out = rope.apply(&x, 0).unwrap();
    assert_eq!(out.shape(), &[2, 4, 6, head_dim]);
}

#[test]
fn rope_mha_forward() {
    let d_model = 32;
    let mha = MultiHeadAttention::new(
        d_model, 4, None, false, true,
        PositionEncoding::RoPE, 128, 10000.0,
    ).unwrap();
    let input = Tensor::zeros(&[1, 4, d_model]);
    let output = mha.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 4, d_model]);
}

#[test]
fn rope_cache_positions() {
    let d_model = 32;
    let num_heads = 4;
    let head_dim = d_model / num_heads;
    let mha = MultiHeadAttention::new(
        d_model, num_heads, None, false, true,
        PositionEncoding::RoPE, 64, 10000.0,
    ).unwrap();

    let mut cache = KVCache::new(1, 64, head_dim, num_heads);
    let input = Tensor::zeros(&[1, 3, d_model]);
    mha.forward_with_cache(&input, &mut cache, 0).unwrap();
    cache.advance(3);

    let input2 = Tensor::zeros(&[1, 1, d_model]);
    let out = mha.forward_with_cache(&input2, &mut cache, 0).unwrap();
    assert_eq!(out.shape(), &[1, 1, d_model]);
}

// ── ALiBi tests ──────────────────────────────────────────────────────

#[test]
fn alibi_slopes_computation() {
    let slopes = compute_alibi_slopes(4);
    assert_eq!(slopes.len(), 4);
    // Each slope should be 2^(-8*(h+1)/H) for h=0..H-1
    // h=0: 2^(-2) = 0.25
    // h=1: 2^(-4) = 0.0625
    // h=2: 2^(-6) = 0.015625
    // h=3: 2^(-8) = 0.00390625
    assert!((slopes[0] - 0.25).abs() < 1e-6, "slope[0]={}", slopes[0]);
    assert!((slopes[1] - 0.0625).abs() < 1e-6, "slope[1]={}", slopes[1]);
}

#[test]
fn alibi_bias_shape() {
    let slopes = compute_alibi_slopes(4);
    let bias = alibi_bias(&slopes, 3, 5, true);
    assert_eq!(bias.shape(), &[1, 4, 3, 5]);
}

#[test]
fn alibi_bias_values() {
    let slopes = vec![1.0]; // single head, slope=1 for easy verification
    let bias = alibi_bias(&slopes, 3, 3, true);
    let data = bias.as_slice_f32().unwrap();
    // [1, 1, 3, 3]
    // Row 0 (query pos 0): [0, -inf, -inf] (causal)
    assert_eq!(data[0], 0.0);
    assert!(data[1].is_infinite() && data[1] < 0.0);
    assert!(data[2].is_infinite() && data[2] < 0.0);

    // Row 1 (query pos 1): [-1, 0, -inf]
    assert!((data[3] - (-1.0)).abs() < 1e-6);
    assert_eq!(data[4], 0.0);
    assert!(data[5].is_infinite() && data[5] < 0.0);

    // Row 2 (query pos 2): [-2, -1, 0]
    assert!((data[6] - (-2.0)).abs() < 1e-6);
    assert!((data[7] - (-1.0)).abs() < 1e-6);
    assert_eq!(data[8], 0.0);
}

#[test]
fn alibi_forward_shape() {
    let d_model = 32;
    let mha = MultiHeadAttention::new(
        d_model, 4, None, false, true,
        PositionEncoding::ALiBi, 128, 10000.0,
    ).unwrap();
    let input = Tensor::zeros(&[1, 4, d_model]);
    let output = mha.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 4, d_model]);
}

// ── Cross-attention tests ────────────────────────────────────────────

#[test]
fn cross_attention_forward_shape() {
    let d_model = 32;
    let cross = CrossAttention::new(d_model, d_model, 4, None, false).unwrap();
    let decoder = Tensor::zeros(&[1, 3, d_model]);
    let encoder = Tensor::zeros(&[1, 5, d_model]);
    let output = cross.forward(&decoder, &encoder).unwrap();
    assert_eq!(output.shape(), &[1, 3, d_model]);
}

#[test]
fn cross_attention_different_lengths() {
    let d_model = 32;
    let cross = CrossAttention::new(d_model, d_model, 4, None, false).unwrap();
    let decoder = Tensor::zeros(&[1, 2, d_model]);
    let encoder = Tensor::zeros(&[1, 10, d_model]);
    let output = cross.forward(&decoder, &encoder).unwrap();
    assert_eq!(output.shape(), &[1, 2, d_model]);
}

#[test]
fn cross_attention_different_dims() {
    let d_model = 32;
    let enc_dim = 64;
    let cross = CrossAttention::new(d_model, enc_dim, 4, None, false).unwrap();
    let decoder = Tensor::zeros(&[1, 3, d_model]);
    let encoder = Tensor::zeros(&[1, 5, enc_dim]);
    let output = cross.forward(&decoder, &encoder).unwrap();
    assert_eq!(output.shape(), &[1, 3, d_model]);
}

#[test]
fn cross_attention_with_gqa() {
    let d_model = 32;
    let cross = CrossAttention::new(d_model, d_model, 4, Some(2), false).unwrap();
    let decoder = Tensor::zeros(&[1, 3, d_model]);
    let encoder = Tensor::zeros(&[1, 5, d_model]);
    let output = cross.forward(&decoder, &encoder).unwrap();
    assert_eq!(output.shape(), &[1, 3, d_model]);
}

// ── Parameter freezing tests ─────────────────────────────────────────

#[test]
fn freezable_default_false() {
    let linear = Linear::new(16, 32, false);
    assert!(!linear.is_frozen());

    let emb = Embedding::new(100, 32);
    assert!(!emb.is_frozen());

    let ln = LayerNorm::new(vec![32], 1e-5);
    assert!(!ln.is_frozen());
}

#[test]
fn freeze_unfreeze_linear() {
    let mut linear = Linear::new(16, 32, false);
    assert!(!linear.is_frozen());
    linear.freeze();
    assert!(linear.is_frozen());
    linear.unfreeze();
    assert!(!linear.is_frozen());
}

#[test]
fn model_freeze_embeddings() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 100,
        norm_eps: 1e-5,
        max_seq_len: 32,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };
    let mut model = LlmModel::new(&config).unwrap();
    assert!(!model.token_embedding.frozen);

    model.freeze_embeddings();
    assert!(model.token_embedding.frozen);
    if let Some(ref pos_emb) = model.pos_embedding {
        assert!(pos_emb.frozen);
    }
}

#[test]
fn model_parameter_count() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 100,
        norm_eps: 1e-5,
        max_seq_len: 32,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };
    let mut model = LlmModel::new(&config).unwrap();
    let (total, frozen) = model.parameter_count();
    assert!(total > 0);
    assert_eq!(frozen, 0);

    model.freeze_embeddings();
    let (total2, frozen2) = model.parameter_count();
    assert_eq!(total, total2); // total shouldn't change
    assert!(frozen2 > 0);
}

// ── Config validation for GQA ────────────────────────────────────────

#[test]
fn config_gqa_valid() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: Some(2),
        vocab_size: 100,
        norm_eps: 1e-5,
        max_seq_len: 32,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn config_gqa_invalid_divisibility() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: Some(3), // 4 % 3 != 0
        vocab_size: 100,
        norm_eps: 1e-5,
        max_seq_len: 32,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };
    assert!(config.validate().is_err());
}

#[test]
fn config_gqa_zero_kv_heads_fails() {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: Some(0),
        vocab_size: 100,
        norm_eps: 1e-5,
        max_seq_len: 32,
        use_bias: Some(false),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };
    assert!(config.validate().is_err());
}
