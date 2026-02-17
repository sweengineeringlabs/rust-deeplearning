use llmforge::core::tensor::{Tensor, DType, f32_vec_to_bytes};
use llmforge::quantization::{
    quantize_tensor, quantize_tensor_q4, dequantize_tensor,
    quantized_matmul, quantized_matmul_q4, quantized_matmul_q4_native,
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
    let mut packed = vec![0u8; 16];
    for i in 0..16 {
        let lo_val: u8 = (i % 16) as u8;  // [0,15]
        let hi_val: u8 = ((i + 1) % 16) as u8;
        packed[i] = (hi_val << 4) | (lo_val & 0x0F);
    }
    let scale = 0.25;

    let result = simd::dot_q4_block_scalar_ref(&input, &packed, scale);

    // Manual: lo nibbles → positions 0..15, hi nibbles → positions 16..31
    let mut expected = 0.0f32;
    for j in 0..16 {
        let lo = (packed[j] & 0x0F) as i8 - 8;
        let hi = ((packed[j] >> 4) & 0x0F) as i8 - 8;
        expected += input[j] * (lo as f32);
        expected += input[j + 16] * (hi as f32);
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

// --- Native Q4_0 × Q8_0 integer dot product tests ---

#[test]
fn scalar_q4q8_block_correctness() {
    // Known packed bytes: nibbles [3,5,0,15, ...] and i8 values
    let mut packed = [0u8; 16];
    packed[0] = 0x53; // lo=3, hi=5
    packed[1] = 0xF0; // lo=0, hi=15
    for i in 2..16 {
        packed[i] = 0x88; // lo=8(→0), hi=8(→0)
    }

    let mut q8 = [0i8; 32];
    q8[0] = 10;   // pairs with lo nibble 0 → dequant 3-8 = -5
    q8[1] = 20;   // pairs with lo nibble 1 → dequant 0-8 = -8
    q8[16] = 5;   // pairs with hi nibble 0 → dequant 5-8 = -3
    q8[17] = -3;  // pairs with hi nibble 1 → dequant 15-8 = 7

    // Manual: (-5)*10 + (-8)*20 + 0*0... + (-3)*5 + 7*(-3) + 0*0...
    //       = -50 + -160 + -15 + -21 = -246
    let expected: i32 = -246;
    let result = simd::dot_q4q8_block_scalar_ref(&packed, &q8);
    assert_eq!(result, expected, "scalar q4q8: result={}, expected={}", result, expected);
}

#[test]
fn dispatch_matches_scalar_q4q8() {
    // Random-ish values
    let mut packed = [0u8; 16];
    for i in 0..16 {
        packed[i] = (i as u8).wrapping_mul(7).wrapping_add(3);
    }
    let mut q8 = [0i8; 32];
    for i in 0..32 {
        q8[i] = ((i as i32 * 13 - 50) % 128) as i8;
    }

    let scalar = simd::dot_q4q8_block_scalar_ref(&packed, &q8);
    let dispatched = simd::dot_q4q8_block(&packed, &q8);

    // Integer results must match exactly
    assert_eq!(
        scalar, dispatched,
        "scalar={}, dispatched={}", scalar, dispatched
    );
}

#[test]
fn q4q8_all_zeros() {
    let packed = [0u8; 16]; // all nibbles = 0, dequant = -8
    let q8 = [0i8; 32];    // all zeros

    let result = simd::dot_q4q8_block(&packed, &q8);
    assert_eq!(result, 0, "all-zero q8 should produce 0 regardless of q4 values");
}

#[test]
fn q4q8_max_values() {
    // All nibbles = 15 → dequant = 7, all q8 = 127
    let packed = [0xFFu8; 16]; // lo=15, hi=15 for all bytes
    let q8 = [127i8; 32];

    // Expected: 32 × 7 × 127 = 28448
    let result = simd::dot_q4q8_block(&packed, &q8);
    assert_eq!(result, 28448, "max values: result={}, expected=28448", result);
}

#[test]
fn native_q4_matmul_correctness() {
    // Small matrix: input [1, 64], weights [2, 64] in Q4_0
    let input_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    let input = make_f32_tensor(&input_data, vec![1, 64]);

    let weight_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.005).collect();
    let weight_f32 = make_f32_tensor(&weight_data, vec![2, 64]);
    let weight_q4 = quantize_tensor_q4(&weight_f32).unwrap();

    let result = quantized_matmul_q4_native(&input, &weight_q4).unwrap();
    assert_eq!(result.shape(), &[1, 2]);

    let data = result.as_slice_f32().unwrap();
    for &v in data {
        assert!(v.is_finite(), "native Q4 matmul produced non-finite: {}", v);
    }
}

#[test]
fn native_vs_dequant_reasonable_agreement() {
    // Both matmul paths should agree within ~1% relative error
    let input_data: Vec<f32> = (0..128).map(|i| ((i as f32) - 64.0) * 0.02).collect();
    let input = make_f32_tensor(&input_data, vec![1, 128]);

    let weight_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.003).collect();
    let weight_f32 = make_f32_tensor(&weight_data, vec![4, 128]);
    let weight_q4 = quantize_tensor_q4(&weight_f32).unwrap();

    let dequant_result = quantized_matmul_q4(&input, &weight_q4).unwrap();
    let native_result = quantized_matmul_q4_native(&input, &weight_q4).unwrap();

    let dequant_data = dequant_result.as_slice_f32().unwrap();
    let native_data = native_result.as_slice_f32().unwrap();

    assert_eq!(dequant_data.len(), native_data.len());
    for i in 0..dequant_data.len() {
        let d = dequant_data[i];
        let n = native_data[i];
        let rel_err = if d.abs() > 1e-6 {
            (d - n).abs() / d.abs()
        } else {
            (d - n).abs()
        };
        assert!(
            rel_err < 0.02,
            "element {}: dequant={}, native={}, rel_err={:.4}",
            i, d, n, rel_err
        );
    }
}
