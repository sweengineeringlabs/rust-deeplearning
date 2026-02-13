use llmforge::core::tensor::{Tensor, DType};
use half::{bf16, f16};

/// Create an F32 tensor from an f32 slice and shape.
pub fn make_f32_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
    Tensor::new(bytes, shape, DType::F32)
}

/// Create a BF16 tensor from f32 values (converted to BF16 bytes).
pub fn make_bf16_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    let bf16_vals: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
    let bytes: Vec<u8> = bytemuck::cast_slice(&bf16_vals).to_vec();
    Tensor::new(bytes, shape, DType::BF16)
}

/// Create an F16 tensor from f32 values (converted to F16 bytes).
pub fn make_f16_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    let f16_vals: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    let bytes: Vec<u8> = bytemuck::cast_slice(&f16_vals).to_vec();
    Tensor::new(bytes, shape, DType::F16)
}

/// Assert that two f32 slices are element-wise close within a tolerance.
pub fn assert_f32_near(actual: &[f32], expected: &[f32], tolerance: f32, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch (actual={}, expected={})",
        msg,
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "{}: element [{}] mismatch: actual={}, expected={}, diff={}, tolerance={}",
            msg,
            i,
            a,
            e,
            (a - e).abs(),
            tolerance
        );
    }
}
