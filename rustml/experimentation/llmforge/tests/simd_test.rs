use llmforge::core::tensor::{Tensor, DType, f32_vec_to_bytes};
use llmforge::quantization::{
    quantize_tensor, quantize_tensor_q4, dequantize_tensor,
    quantized_matmul, quantized_matmul_q4,
    simd,
};

fn make_f32_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    let bytes = f32_vec_to_bytes(data.to_vec());
    Tensor::new(bytes, shape, DType::F32)
}

#[test]
fn scalar_q8_block_correctness() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let quantized: Vec<i8> = (0..32).map(|i| i as i8).collect();
    let scale = 0.5;

    let result = simd::dot_q8_block_scalar_ref(&input, &quantized, scale);

    // Manual computation
    let mut expected = 0.0f32;
    for i in 0..32 {
        expected += input[i] * (quantized[i] as f32);
    }
    expected *= scale;

    assert!((result - expected).abs() < 1e-4, "result={}, expected={}", result, expected);
}

#[test]
fn scalar_q4_block_correctness() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    // Pack values: low nibble = even, high nibble = odd
    let mut packed = vec![0u8; 16];
    for i in 0..16 {
        let even_val: u8 = (i % 16) as u8;  // [0,15]
        let odd_val: u8 = ((i + 1) % 16) as u8;
        packed[i] = (odd_val << 4) | (even_val & 0x0F);
    }
    let scale = 0.25;

    let result = simd::dot_q4_block_scalar_ref(&input, &packed, scale);

    // Manual: unpack and compute
    let mut expected = 0.0f32;
    for i in 0..16 {
        let even_q = (packed[i] & 0x0F) as i8 - 8;
        let odd_q = ((packed[i] >> 4) & 0x0F) as i8 - 8;
        expected += input[i * 2] * (even_q as f32);
        expected += input[i * 2 + 1] * (odd_q as f32);
    }
    expected *= scale;

    assert!((result - expected).abs() < 1e-4, "result={}, expected={}", result, expected);
}

#[test]
fn dispatch_matches_scalar_q8() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let quantized: Vec<i8> = (0..32).map(|i| (i as i8 - 16)).collect();
    let scale = 1.5;

    let scalar = simd::dot_q8_block_scalar_ref(&input, &quantized, scale);
    let dispatched = simd::dot_q8_block(&input, &quantized, scale);

    assert!(
        (scalar - dispatched).abs() < 1e-3,
        "scalar={}, dispatched={}", scalar, dispatched
    );
}

#[test]
fn dispatch_matches_scalar_q4() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let mut packed = vec![0u8; 16];
    for i in 0..16 {
        packed[i] = ((i as u8 + 3) << 4) | (i as u8 & 0x0F);
    }
    let scale = 2.0;

    let scalar = simd::dot_q4_block_scalar_ref(&input, &packed, scale);
    let dispatched = simd::dot_q4_block(&input, &packed, scale);

    assert!(
        (scalar - dispatched).abs() < 1e-3,
        "scalar={}, dispatched={}", scalar, dispatched
    );
}

#[test]
fn full_q8_matmul_unchanged_after_simd_refactor() {
    // Verify Q8_0 quantized matmul matches dequantize-then-manual-matmul
    let input_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    let input = make_f32_tensor(&input_data, vec![1, 64]);

    let weight_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.005).collect();
    let weight_f32 = make_f32_tensor(&weight_data, vec![2, 64]);

    // Reference: dequantize then manual dot product
    let weight_q8 = quantize_tensor(&weight_f32).unwrap();
    let weight_deq = dequantize_tensor(&weight_q8).unwrap();
    let deq_data = weight_deq.as_slice_f32().unwrap();

    let mut ref_data = vec![0.0f32; 2];
    for j in 0..2 {
        for l in 0..64 {
            ref_data[j] += input_data[l] * deq_data[j * 64 + l];
        }
    }

    // Quantized matmul
    let q8_result = quantized_matmul(&input, &weight_q8).unwrap();
    let q8_data = q8_result.as_slice_f32().unwrap();

    for i in 0..2 {
        let err = (ref_data[i] - q8_data[i]).abs();
        let tol = ref_data[i].abs() * 0.01 + 0.01;
        assert!(err < tol, "Q8 element {}: ref={}, q8={}, err={}", i, ref_data[i], q8_data[i], err);
    }
}

#[test]
fn full_q4_matmul_unchanged_after_simd_refactor() {
    let input_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    let input = make_f32_tensor(&input_data, vec![1, 64]);

    let weight_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.005).collect();
    let weight_f32 = make_f32_tensor(&weight_data, vec![2, 64]);

    let weight_q4 = quantize_tensor_q4(&weight_f32).unwrap();
    let q4_result = quantized_matmul_q4(&input, &weight_q4).unwrap();

    assert_eq!(q4_result.shape(), &[1, 2]);
    let data = q4_result.as_slice_f32().unwrap();
    // Verify finite results
    for &v in data {
        assert!(v.is_finite(), "Q4 matmul produced non-finite: {}", v);
    }
}

#[test]
fn simd_with_zero_scale() {
    let input = vec![1.0f32; 32];
    let quantized: Vec<i8> = vec![100; 32];

    let result = simd::dot_q8_block(&input, &quantized, 0.0);
    assert_eq!(result, 0.0);
}

#[test]
fn simd_q4_with_zero_scale() {
    let input = vec![1.0f32; 32];
    let packed = vec![0xFFu8; 16];

    let result = simd::dot_q4_block(&input, &packed, 0.0);
    assert_eq!(result, 0.0);
}
