use llmforge::core::tensor::{Tensor, DType, f32_vec_to_bytes};
use llmforge::quantization::{
    quantize_tensor_q4, dequantize_tensor_q4, quantized_matmul_q4,
    Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES,
};
use llmforge::nn::{Linear, Layer};

fn make_f32_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    let bytes = f32_vec_to_bytes(data.to_vec());
    Tensor::new(bytes, shape, DType::F32)
}

#[test]
fn q4_0_roundtrip_within_tolerance() {
    // 32 values in [-1, 1] range where Q4_0 should be reasonably accurate
    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
    let tensor = make_f32_tensor(&data, vec![32]);

    let quantized = quantize_tensor_q4(&tensor).unwrap();
    assert_eq!(quantized.dtype(), DType::Q4_0);

    let dequantized = dequantize_tensor_q4(&quantized).unwrap();
    assert_eq!(dequantized.dtype(), DType::F32);

    let orig = tensor.as_slice_f32().unwrap();
    let result = dequantized.as_slice_f32().unwrap();

    for (i, (a, b)) in orig.iter().zip(result.iter()).enumerate() {
        let err = (a - b).abs();
        // Q4_0 has much lower precision than Q8_0, allow larger tolerance
        assert!(err < 0.25, "Element {}: {} vs {}, err={}", i, a, b, err);
    }
}

#[test]
fn q4_0_produces_correct_dtype() {
    let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let tensor = make_f32_tensor(&data, vec![2, 32]);
    let quantized = quantize_tensor_q4(&tensor).unwrap();
    assert_eq!(quantized.dtype(), DType::Q4_0);
    assert_eq!(quantized.shape(), &[2, 32]);
}

#[test]
fn q4_0_memory_savings() {
    // 128 F32 elements = 512 bytes
    let data = vec![1.0f32; 128];
    let tensor = make_f32_tensor(&data, vec![128]);
    let f32_bytes = tensor.as_raw_bytes().unwrap().len();

    let quantized = quantize_tensor_q4(&tensor).unwrap();
    let q4_bytes = quantized.as_raw_bytes().unwrap().len();

    // Q4_0: 128/32 = 4 blocks * 18 bytes = 72 bytes
    assert_eq!(q4_bytes, 72);

    // Should be ~14% of F32 (72/512 = 14.06%)
    let ratio = q4_bytes as f64 / f32_bytes as f64;
    assert!(ratio < 0.15, "Memory ratio {} should be ~14%", ratio);
    assert!(ratio > 0.13, "Memory ratio {} should be ~14%", ratio);
}

#[test]
fn q4_0_block_size_validation() {
    // Not divisible by 32 should error
    let data = vec![1.0f32; 16];
    let tensor = make_f32_tensor(&data, vec![16]);
    let result = quantize_tensor_q4(&tensor);
    assert!(result.is_err());
}

#[test]
fn q4_0_requires_f32_input() {
    let data = vec![0u8; 34]; // Q8_0 dummy
    let tensor = Tensor::new(data, vec![32], DType::Q8_0);
    assert!(quantize_tensor_q4(&tensor).is_err());
}

#[test]
fn q4_0_dequantize_requires_q4_dtype() {
    let data = vec![1.0f32; 32];
    let tensor = make_f32_tensor(&data, vec![32]);
    assert!(dequantize_tensor_q4(&tensor).is_err());
}

#[test]
fn q4_0_matmul_correctness() {
    // Simple matmul: [1, 64] x [2, 64]^T = [1, 2]
    let input_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    let input = make_f32_tensor(&input_data, vec![1, 64]);

    // Weight [2, 64] with known values
    let mut weight_data = Vec::new();
    for _ in 0..2 {
        for j in 0..64 {
            weight_data.push((j as f32) * 0.02);
        }
    }
    let weight_f32 = make_f32_tensor(&weight_data, vec![2, 64]);

    // Get reference result via standard matmul
    let w_t = weight_f32.transpose(0, 1).unwrap();
    let ref_result = input.matmul(&w_t).unwrap();
    let ref_data = ref_result.as_slice_f32().unwrap();

    // Q4_0 quantized matmul
    let weight_q4 = quantize_tensor_q4(&weight_f32).unwrap();
    let q4_result = quantized_matmul_q4(&input, &weight_q4).unwrap();
    let q4_data = q4_result.as_slice_f32().unwrap();

    assert_eq!(q4_result.shape(), &[1, 2]);

    for i in 0..2 {
        let rel_err = if ref_data[i].abs() > 1e-6 {
            (ref_data[i] - q4_data[i]).abs() / ref_data[i].abs()
        } else {
            (ref_data[i] - q4_data[i]).abs()
        };
        // Q4_0 allows larger relative error due to 4-bit precision
        assert!(rel_err < 0.2, "Element {}: ref={}, q4={}, rel_err={}", i, ref_data[i], q4_data[i], rel_err);
    }
}

#[test]
fn q4_0_linear_integration() {
    let mut linear = Linear::new(64, 16, false);
    assert!(!linear.is_quantized());

    linear.quantize_q4().unwrap();
    assert!(linear.is_quantized());
    assert_eq!(linear.weight.dtype(), DType::Q4_0);

    // Forward should work
    let input_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    let input = make_f32_tensor(&input_data, vec![1, 64]);
    let output = linear.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 16]);
}

#[test]
fn q4_0_3d_input_matmul() {
    // [2, 3, 64] x [16, 64]^T = [2, 3, 16]
    let input_data: Vec<f32> = (0..2 * 3 * 64).map(|i| (i as f32) * 0.001).collect();
    let input = make_f32_tensor(&input_data, vec![2, 3, 64]);

    let weight_data: Vec<f32> = (0..16 * 64).map(|i| (i as f32) * 0.001).collect();
    let weight_f32 = make_f32_tensor(&weight_data, vec![16, 64]);
    let weight_q4 = quantize_tensor_q4(&weight_f32).unwrap();

    let result = quantized_matmul_q4(&input, &weight_q4).unwrap();
    assert_eq!(result.shape(), &[2, 3, 16]);
}

#[test]
fn q4_0_zeros_quantize_correctly() {
    let data = vec![0.0f32; 32];
    let tensor = make_f32_tensor(&data, vec![32]);

    let quantized = quantize_tensor_q4(&tensor).unwrap();
    let dequantized = dequantize_tensor_q4(&quantized).unwrap();
    let result = dequantized.as_slice_f32().unwrap();

    for &v in result {
        assert_eq!(v, 0.0);
    }
}
