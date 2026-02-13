mod common;

use common::{make_f32_tensor, make_bf16_tensor, make_f16_tensor, assert_f32_near};
use llmforge::core::tensor::{Tensor, DType};

#[test]
fn f32_to_bf16_to_f32_roundtrip() {
    let original = [1.0f32, -2.5, 3.14, 0.0, 100.0, -0.001];
    let bf16_tensor = make_bf16_tensor(&original, vec![2, 3]);
    assert_eq!(bf16_tensor.dtype(), DType::BF16);

    let f32_tensor = bf16_tensor.to_f32().unwrap();
    assert_eq!(f32_tensor.dtype(), DType::F32);
    assert_eq!(f32_tensor.shape(), &[2, 3]);

    let data = f32_tensor.as_slice_f32().unwrap();
    // BF16 has ~0.01 relative precision for values near 1
    assert_f32_near(data, &[1.0, -2.5, 3.14, 0.0, 100.0, -0.001], 0.05, "bf16 roundtrip");
}

#[test]
fn f32_to_f16_to_f32_roundtrip() {
    let original = [1.0f32, -2.5, 3.14, 0.0, 100.0, -0.001];
    let f16_tensor = make_f16_tensor(&original, vec![2, 3]);
    assert_eq!(f16_tensor.dtype(), DType::F16);

    let f32_tensor = f16_tensor.to_f32().unwrap();
    assert_eq!(f32_tensor.dtype(), DType::F32);
    assert_eq!(f32_tensor.shape(), &[2, 3]);

    let data = f32_tensor.as_slice_f32().unwrap();
    // F16 has ~0.001 relative precision
    assert_f32_near(data, &[1.0, -2.5, 3.14, 0.0, 100.0, -0.001], 0.01, "f16 roundtrip");
}

#[test]
fn to_f32_on_f32_is_identity() {
    let original = [1.0f32, 2.0, 3.0, 4.0];
    let t = make_f32_tensor(&original, vec![4]);
    let converted = t.to_f32().unwrap();
    assert_eq!(converted.dtype(), DType::F32);
    let data = converted.as_slice_f32().unwrap();
    assert_f32_near(data, &original, 0.0, "f32 to f32 identity");
}

#[test]
fn to_f32_unsupported_dtype_errors() {
    // I8 tensor -> to_f32() should return NotImplemented
    let bytes = vec![1u8, 2, 3, 4];
    let t = Tensor::new(bytes, vec![4], DType::I8);
    let result = t.to_f32();
    assert!(result.is_err(), "Expected error for I8 -> F32 conversion");
}
