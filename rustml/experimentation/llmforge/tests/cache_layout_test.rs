use llmforge::core::tensor::{Tensor, DType, f32_vec_to_bytes};
use llmforge::quantization::{
    quantize_tensor, quantize_tensor_q4, dequantize_tensor, dequantize_tensor_q4,
    quantized_matmul, quantized_matmul_q4,
};

fn make_f32_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    let bytes = f32_vec_to_bytes(data.to_vec());
    Tensor::new(bytes, shape, DType::F32)
}

/// Reference: dequantize weights, then compute manual dot products.
fn manual_matmul_f32(input: &[f32], weights: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut dot = 0.0f32;
            for l in 0..k {
                dot += input[i * k + l] * weights[j * k + l];
            }
            out[i * n + j] = dot;
        }
    }
    out
}

#[test]
fn tiled_q8_matmul_matches_dequantized_reference() {
    let m = 4;
    let k = 128;
    let n = 32;

    let input_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let input = make_f32_tensor(&input_data, vec![m, k]);

    let weight_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.005).collect();
    let weight_f32 = make_f32_tensor(&weight_data, vec![n, k]);

    // Quantize then dequantize to get the reference (what Q8 "sees")
    let weight_q8 = quantize_tensor(&weight_f32).unwrap();
    let weight_deq = dequantize_tensor(&weight_q8).unwrap();
    let deq_data = weight_deq.as_slice_f32().unwrap();

    // Manual reference matmul with dequantized weights
    let ref_data = manual_matmul_f32(&input_data, deq_data, m, k, n);

    // On-the-fly quantized matmul
    let q8_result = quantized_matmul(&input, &weight_q8).unwrap();
    let q8_data = q8_result.as_slice_f32().unwrap();

    assert_eq!(q8_result.shape(), &[m, n]);

    for i in 0..(m * n) {
        let err = (ref_data[i] - q8_data[i]).abs();
        let tol = ref_data[i].abs() * 0.01 + 0.1; // 1% relative + 0.1 absolute
        assert!(err < tol, "Element {}: ref={}, q8={}, err={}", i, ref_data[i], q8_data[i], err);
    }
}

#[test]
fn tiled_q4_matmul_matches_dequantized_reference() {
    let m = 4;
    let k = 128;
    let n = 32;

    let input_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let input = make_f32_tensor(&input_data, vec![m, k]);

    let weight_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.005).collect();
    let weight_f32 = make_f32_tensor(&weight_data, vec![n, k]);

    // Quantize then dequantize for reference
    let weight_q4 = quantize_tensor_q4(&weight_f32).unwrap();
    let weight_deq = dequantize_tensor_q4(&weight_q4).unwrap();
    let deq_data = weight_deq.as_slice_f32().unwrap();

    // Manual reference
    let ref_data = manual_matmul_f32(&input_data, deq_data, m, k, n);

    // On-the-fly quantized matmul
    let q4_result = quantized_matmul_q4(&input, &weight_q4).unwrap();
    let q4_data = q4_result.as_slice_f32().unwrap();

    assert_eq!(q4_result.shape(), &[m, n]);

    for i in 0..(m * n) {
        let err = (ref_data[i] - q4_data[i]).abs();
        let tol = ref_data[i].abs() * 0.02 + 0.5; // 2% relative + 0.5 absolute
        assert!(err < tol, "Element {}: ref={}, q4={}, err={}", i, ref_data[i], q4_data[i], err);
    }
}

#[test]
fn alignment_verification() {
    let tensor = Tensor::new_aligned(vec![4, 8], DType::F32);
    assert_eq!(tensor.shape(), &[4, 8]);
    assert_eq!(tensor.dtype(), DType::F32);

    let bytes = tensor.as_raw_bytes().unwrap();
    assert_eq!(bytes.len(), 4 * 8 * 4); // 4*8 f32 elements * 4 bytes
}

#[test]
fn large_matrix_sanity() {
    // Non-tile-aligned dimensions to exercise tiling edge cases
    let m = 7;
    let k = 96;
    let n = 13;

    let input_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
    let input = make_f32_tensor(&input_data, vec![m, k]);

    let weight_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.002).collect();
    let weight_f32 = make_f32_tensor(&weight_data, vec![n, k]);

    let weight_q8 = quantize_tensor(&weight_f32).unwrap();
    let result = quantized_matmul(&input, &weight_q8).unwrap();
    assert_eq!(result.shape(), &[m, n]);

    let weight_q4 = quantize_tensor_q4(&weight_f32).unwrap();
    let result_q4 = quantized_matmul_q4(&input, &weight_q4).unwrap();
    assert_eq!(result_q4.shape(), &[m, n]);

    for &v in result.as_slice_f32().unwrap() {
        assert!(v.is_finite());
    }
    for &v in result_q4.as_slice_f32().unwrap() {
        assert!(v.is_finite());
    }
}
