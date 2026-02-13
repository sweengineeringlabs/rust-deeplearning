use llmforge::core::tensor::{Tensor, DType};
use llmforge::quantization::{quantize_tensor, dequantize_tensor, quantized_matmul, Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES};
use llmforge::nn::{Linear, Layer};

fn make_f32_tensor(data: &[f32], shape: &[usize]) -> Tensor {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    Tensor::new(bytes, shape.to_vec(), DType::F32)
}

#[test]
fn quantize_produces_q8_0_dtype() {
    // 64 elements = 2 blocks of 32
    let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let tensor = make_f32_tensor(&data, &[2, 32]);

    let quantized = quantize_tensor(&tensor).unwrap();
    assert_eq!(quantized.dtype(), DType::Q8_0);
    assert_eq!(quantized.shape(), &[2, 32]);
}

#[test]
fn quantize_dequantize_roundtrip_within_tolerance() {
    let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.5).collect();
    let tensor = make_f32_tensor(&data, &[4, 32]);

    let quantized = quantize_tensor(&tensor).unwrap();
    let dequantized = dequantize_tensor(&quantized).unwrap();

    assert_eq!(dequantized.dtype(), DType::F32);
    assert_eq!(dequantized.shape(), &[4, 32]);

    let original = tensor.as_slice_f32().unwrap();
    let restored = dequantized.as_slice_f32().unwrap();

    // Find max absolute value for error bound
    let amax = original.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let max_error = amax / 127.0;

    for (i, (&orig, &rest)) in original.iter().zip(restored.iter()).enumerate() {
        let err = (orig - rest).abs();
        assert!(
            err <= max_error + 1e-6,
            "Element {} error {} exceeds tolerance {} (orig={}, restored={})",
            i, err, max_error, orig, rest
        );
    }
}

#[test]
fn quantized_matmul_matches_standard_matmul() {
    // input: [2, 64], weights: [32, 64]
    // Standard: input * W^T = [2, 32]
    let input_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let weight_data: Vec<f32> = (0..2048).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();

    let input = make_f32_tensor(&input_data, &[2, 64]);
    let weights_f32 = make_f32_tensor(&weight_data, &[32, 64]);

    // Standard matmul: input * W^T
    // Must make w_t contiguous — matmul doesn't handle non-contiguous strides for 2D
    let w_t = weights_f32.transpose(0, 1).unwrap().contiguous().unwrap();
    let expected = input.matmul(&w_t).unwrap();

    // Quantized matmul
    let weights_q8 = quantize_tensor(&weights_f32).unwrap();
    let actual = quantized_matmul(&input, &weights_q8).unwrap();

    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.shape(), &[2, 32]);

    let exp_data = expected.as_slice_f32().unwrap();
    let act_data = actual.as_slice_f32().unwrap();

    // Quantization error accumulates across the dot product (64 elements).
    // Use generous tolerance: errors scale with both input magnitude and weight range.
    let max_err = exp_data.iter().zip(act_data.iter())
        .map(|(&e, &a)| (e - a).abs())
        .fold(0.0f32, f32::max);
    let exp_magnitude = exp_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    // Allow up to 20% relative error or 2.0 absolute, whichever is larger
    let tolerance = (exp_magnitude * 0.2).max(2.0);
    assert!(
        max_err < tolerance,
        "Max error {} exceeds tolerance {} (max expected magnitude={})",
        max_err, tolerance, exp_magnitude
    );
}

#[test]
fn linear_quantize_and_forward_correct_shape() {
    let mut linear = Linear::new(64, 32, false);
    assert!(!linear.is_quantized());

    linear.quantize().unwrap();
    assert!(linear.is_quantized());

    // Forward with [2, 64] input
    let input = Tensor::zeros(&[2, 64]);
    let output = linear.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 32]);
}

#[test]
fn linear_quantize_forward_3d_input() {
    let mut linear = Linear::new(64, 32, false);
    linear.quantize().unwrap();

    // Forward with [2, 4, 64] input (batched)
    let input = Tensor::zeros(&[2, 4, 64]);
    let output = linear.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 4, 32]);
}

#[test]
fn q8_0_memory_savings() {
    // 32 F32 elements = 128 bytes
    // Q8_0: 1 block = 34 bytes
    // Ratio: 34/128 ≈ 0.266
    let n_elements = 1024; // 32 blocks
    let f32_bytes = n_elements * 4;
    let q8_bytes = (n_elements / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;

    let ratio = q8_bytes as f64 / f32_bytes as f64;
    assert!(
        (ratio - 34.0 / 128.0).abs() < 0.001,
        "Q8_0 ratio {} != expected {}", ratio, 34.0 / 128.0
    );
}

#[test]
fn quantize_requires_divisible_by_block_size() {
    // 30 elements is not divisible by 32
    let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let tensor = make_f32_tensor(&data, &[30]);

    let result = quantize_tensor(&tensor);
    assert!(result.is_err());
}
