use llmforge::core::tensor::{Tensor, DType};
use llmforge::attention::{MultiHeadAttention, KVCache};

#[test]
fn test_mha_forward() {
    let batch_size = 2;
    let seq_len = 10;
    let d_model = 64;
    let num_heads = 4;

    // Create MHA
    let mha = MultiHeadAttention::new(d_model, num_heads, false).unwrap();

    // Create random input: [Batch, Seq, D_model]
    let input = Tensor::new(
        vec![0u8; batch_size * seq_len * d_model * 4],
        vec![batch_size, seq_len, d_model],
        DType::F32
    );

    // Forward pass
    let output = mha.forward(&input).expect("Forward pass failed");

    // Check output shape
    assert_eq!(output.shape(), &[batch_size, seq_len, d_model]);

    println!("MHA forward successful. Output shape: {:?}", output.shape());
}

#[test]
fn attention_output_shape_with_cache() {
    let d_model = 32;
    let num_heads = 4;
    let head_dim = d_model / num_heads;
    let max_seq_len = 16;

    let mha = MultiHeadAttention::new(d_model, num_heads, false).unwrap();
    let mut cache = KVCache::new(1, max_seq_len, head_dim, num_heads);

    // Prefill with seq_len=3
    let input = Tensor::new(
        vec![0u8; 1 * 3 * d_model * 4],
        vec![1, 3, d_model],
        DType::F32,
    );
    let out1 = mha.forward_with_cache(&input, &mut cache, 0).unwrap();
    assert_eq!(out1.shape(), &[1, 3, d_model]);
    cache.advance(3);

    // Decode single token
    let input2 = Tensor::new(
        vec![0u8; 1 * 1 * d_model * 4],
        vec![1, 1, d_model],
        DType::F32,
    );
    let out2 = mha.forward_with_cache(&input2, &mut cache, 0).unwrap();
    assert_eq!(out2.shape(), &[1, 1, d_model]);
}

#[test]
fn attention_deterministic_with_known_weights() {
    let d_model = 32;
    let num_heads = 4;

    let mha = MultiHeadAttention::new(d_model, num_heads, false).unwrap();

    let input = Tensor::new(
        vec![0u8; 1 * 2 * d_model * 4],
        vec![1, 2, d_model],
        DType::F32,
    );

    // Same input, same weights -> same output
    let out1 = mha.forward(&input).unwrap();
    let out2 = mha.forward(&input).unwrap();

    let d1 = out1.as_slice_f32().unwrap();
    let d2 = out2.as_slice_f32().unwrap();
    assert_eq!(d1.len(), d2.len());
    for (a, b) in d1.iter().zip(d2.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Determinism violated: {} != {}",
            a,
            b
        );
    }
}

#[test]
fn kvcache_update_accumulates_correctly() {
    let num_layers = 2;
    let max_seq = 8;
    let head_dim = 4;
    let num_heads = 2;

    let mut cache = KVCache::new(num_layers, max_seq, head_dim, num_heads);
    assert_eq!(cache.current_len, 0);

    // First update: seq_len=3
    let k1 = Tensor::zeros(&[1, num_heads, 3, head_dim]);
    let v1 = Tensor::zeros(&[1, num_heads, 3, head_dim]);
    cache.update(0, k1, v1).unwrap();
    cache.advance(3);
    assert_eq!(cache.current_len, 3);

    // Second update: seq_len=2
    let k2 = Tensor::zeros(&[1, num_heads, 2, head_dim]);
    let v2 = Tensor::zeros(&[1, num_heads, 2, head_dim]);
    cache.update(0, k2, v2).unwrap();
    cache.advance(2);
    assert_eq!(cache.current_len, 5);

    // Can get a view of the accumulated cache
    let (k_view, v_view) = cache.get_view(0, 5).unwrap();
    assert_eq!(k_view.shape(), &[1, num_heads, 5, head_dim]);
    assert_eq!(v_view.shape(), &[1, num_heads, 5, head_dim]);
}

#[test]
fn kvcache_overflow_returns_error() {
    let mut cache = KVCache::new(1, 4, 8, 2);
    cache.advance(3);

    // Try to add seq_len=2, which would make 3+2=5 > max_seq_len=4
    let k = Tensor::zeros(&[1, 2, 2, 8]);
    let v = Tensor::zeros(&[1, 2, 2, 8]);
    let result = cache.update(0, k, v);
    assert!(
        result.is_err(),
        "Expected SequenceLengthExceeded error for cache overflow"
    );
}

#[test]
fn kvcache_head_dim_mismatch_detected() {
    let d_model = 32;
    let num_heads = 4;
    let _head_dim = d_model / num_heads; // 8

    let mha = MultiHeadAttention::new(d_model, num_heads, false).unwrap();

    // Cache with wrong head_dim (16 instead of 8)
    let mut cache = KVCache::new(1, 16, 16, num_heads);

    let input = Tensor::new(
        vec![0u8; 1 * 1 * d_model * 4],
        vec![1, 1, d_model],
        DType::F32,
    );

    let result = mha.forward_with_cache(&input, &mut cache, 0);
    assert!(
        result.is_err(),
        "Expected error for head_dim mismatch between MHA and cache"
    );
}

#[test]
fn multi_head_splits_dimension_correctly() {
    // d_model=32, num_heads=4 -> head_dim=8
    let d_model = 32;
    let num_heads = 4;
    let mha = MultiHeadAttention::new(d_model, num_heads, false).unwrap();

    // Verify through a forward pass (if head splitting were wrong, forward would fail)
    let input = Tensor::new(
        vec![0u8; 1 * 2 * d_model * 4],
        vec![1, 2, d_model],
        DType::F32,
    );
    let output = mha.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 2, d_model]);

    // d_model=7, num_heads=3 -> should error (7 % 3 != 0)
    let result = MultiHeadAttention::new(7, 3, false);
    assert!(result.is_err(), "Expected error for non-divisible d_model/num_heads");
}
