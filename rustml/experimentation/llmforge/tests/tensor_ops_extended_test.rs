use llmforge::core::tensor::{Tensor, DType};

fn make_tensor(data: &[f32], shape: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
    Tensor::new(bytes, shape, DType::F32)
}

// ── mul tests ────────────────────────────────────────────────────────

#[test]
fn test_mul_same_shape() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = make_tensor(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = a.mul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    let data = c.as_slice_f32().unwrap();
    assert_eq!(data, &[5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn test_mul_broadcast() {
    let a = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = make_tensor(&[10.0, 20.0, 30.0], vec![3]);
    let c = a.mul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    let data = c.as_slice_f32().unwrap();
    assert_eq!(data, &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
}

#[test]
fn test_mul_shape_mismatch() {
    let a = make_tensor(&[1.0, 2.0, 3.0], vec![3]);
    let b = make_tensor(&[1.0, 2.0], vec![2]);
    assert!(a.mul(&b).is_err());
}

// ── cos tests ────────────────────────────────────────────────────────

#[test]
fn test_cos_known_values() {
    let a = make_tensor(&[0.0, std::f32::consts::PI, std::f32::consts::FRAC_PI_2], vec![3]);
    let c = a.cos().unwrap();
    let data = c.as_slice_f32().unwrap();
    assert!((data[0] - 1.0).abs() < 1e-6, "cos(0) = 1");
    assert!((data[1] - (-1.0)).abs() < 1e-6, "cos(pi) = -1");
    assert!((data[2]).abs() < 1e-6, "cos(pi/2) = 0");
}

#[test]
fn test_cos_preserves_shape() {
    let a = make_tensor(&[0.0; 12], vec![3, 4]);
    let c = a.cos().unwrap();
    assert_eq!(c.shape(), &[3, 4]);
}

// ── sin tests ────────────────────────────────────────────────────────

#[test]
fn test_sin_known_values() {
    let a = make_tensor(&[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI], vec![3]);
    let s = a.sin().unwrap();
    let data = s.as_slice_f32().unwrap();
    assert!((data[0]).abs() < 1e-6, "sin(0) = 0");
    assert!((data[1] - 1.0).abs() < 1e-6, "sin(pi/2) = 1");
    assert!((data[2]).abs() < 1e-5, "sin(pi) ≈ 0");
}

// ── neg tests ────────────────────────────────────────────────────────

#[test]
fn test_neg_values() {
    let a = make_tensor(&[1.0, -2.0, 0.0, 3.5], vec![4]);
    let n = a.neg().unwrap();
    let data = n.as_slice_f32().unwrap();
    assert_eq!(data, &[-1.0, 2.0, 0.0, -3.5]);
}

// ── mul_scalar tests ─────────────────────────────────────────────────

#[test]
fn test_mul_scalar_values() {
    let a = make_tensor(&[1.0, 2.0, 3.0], vec![3]);
    let c = a.mul_scalar(2.5).unwrap();
    let data = c.as_slice_f32().unwrap();
    assert!((data[0] - 2.5).abs() < 1e-6);
    assert!((data[1] - 5.0).abs() < 1e-6);
    assert!((data[2] - 7.5).abs() < 1e-6);
}

#[test]
fn test_mul_scalar_zero() {
    let a = make_tensor(&[1.0, 2.0, 3.0], vec![3]);
    let c = a.mul_scalar(0.0).unwrap();
    let data = c.as_slice_f32().unwrap();
    assert_eq!(data, &[0.0, 0.0, 0.0]);
}

// ── repeat_kv tests ──────────────────────────────────────────────────

#[test]
fn test_repeat_kv_noop() {
    let t = Tensor::zeros(&[1, 2, 3, 4]);
    let r = t.repeat_kv(1).unwrap();
    assert_eq!(r.shape(), &[1, 2, 3, 4]);
}

#[test]
fn test_repeat_kv_2x() {
    // [1, 2, 1, 4] -> [1, 4, 1, 4]
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let t = make_tensor(&data, vec![1, 2, 1, 4]);
    let r = t.repeat_kv(2).unwrap();
    assert_eq!(r.shape(), &[1, 4, 1, 4]);
    let out = r.as_slice_f32().unwrap();
    // Head 0: [0,1,2,3], repeated: [0,1,2,3], [0,1,2,3]
    // Head 1: [4,5,6,7], repeated: [4,5,6,7], [4,5,6,7]
    assert_eq!(&out[0..4], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(&out[4..8], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(&out[8..12], &[4.0, 5.0, 6.0, 7.0]);
    assert_eq!(&out[12..16], &[4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn test_repeat_kv_wrong_dims() {
    let t = Tensor::zeros(&[2, 3]);
    assert!(t.repeat_kv(2).is_err());
}

// ── causal_mask tests ────────────────────────────────────────────────

#[test]
fn test_causal_mask_shape() {
    let mask = Tensor::causal_mask(3, 5);
    assert_eq!(mask.shape(), &[1, 1, 3, 5]);
}

#[test]
fn test_causal_mask_values() {
    let mask = Tensor::causal_mask(3, 3);
    let data = mask.as_slice_f32().unwrap();
    // Row 0: [0, -inf, -inf]
    // Row 1: [0, 0, -inf]
    // Row 2: [0, 0, 0]
    assert_eq!(data[0], 0.0);
    assert!(data[1].is_infinite() && data[1] < 0.0);
    assert!(data[2].is_infinite() && data[2] < 0.0);
    assert_eq!(data[3], 0.0);
    assert_eq!(data[4], 0.0);
    assert!(data[5].is_infinite() && data[5] < 0.0);
    assert_eq!(data[6], 0.0);
    assert_eq!(data[7], 0.0);
    assert_eq!(data[8], 0.0);
}

#[test]
fn test_causal_mask_offset() {
    // seq_len=1, total_len=4: decode step after 3 cached tokens
    // Should attend to all 4 positions (offset=3, i=0, so j <= 0+3=3 is all positions)
    let mask = Tensor::causal_mask(1, 4);
    let data = mask.as_slice_f32().unwrap();
    assert_eq!(data.len(), 4);
    // All positions should be 0 (attend to entire past + current)
    for &v in data {
        assert_eq!(v, 0.0);
    }
}
