//! RMSNorm: x * weight / rms(x). Used by Llama-family models.
//! Unlike LayerNorm, does not subtract mean and has no bias parameter.

use crate::api::error::NnResult;
use crate::api::traits::Freezable;
use rustml_core::Tensor;

/// RMSNorm layer with optional weight offset (used by Gemma: weight + 1.0).
#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
    pub offset: f32,
    pub frozen: bool,
}

impl RMSNorm {
    /// Create a new RMSNorm with weights initialized to 1.
    pub fn new(dim: usize, eps: f32) -> Self {
        let weight = Tensor::ones(vec![dim]);
        Self { weight, eps, offset: 0.0, frozen: false }
    }

    /// Create from a pre-loaded weight tensor.
    pub fn from_weight(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps, offset: 0.0, frozen: false }
    }

    /// Create from a pre-loaded weight tensor with an additive offset.
    ///
    /// Gemma models use `offset = 1.0` so the effective weight is `w + 1.0`.
    pub fn from_weight_with_offset(weight: Tensor, eps: f32, offset: f32) -> Self {
        Self { weight, eps, offset, frozen: false }
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let total = self.weight.numel();
        let frozen = if self.frozen { total } else { 0 };
        (total, frozen)
    }

    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        if self.offset == 0.0 {
            Ok(x.rms_norm(&self.weight, self.eps)?)
        } else {
            let w = self.weight.add_scalar(self.offset);
            Ok(x.rms_norm(&w, self.eps)?)
        }
    }

    /// In-place RMSNorm: overwrites `x` with `rms_norm(x, weight, eps)`.
    ///
    /// Avoids an output allocation when the caller owns the tensor uniquely
    /// (Arc refcount == 1). Falls back to a copy otherwise.
    pub fn forward_inplace(&self, x: &mut Tensor) -> NnResult<()> {
        if self.offset == 0.0 {
            Ok(x.rms_norm_inplace(&self.weight, self.eps)?)
        } else {
            let w = self.weight.add_scalar(self.offset);
            Ok(x.rms_norm_inplace(&w, self.eps)?)
        }
    }
}

impl Freezable for RMSNorm {
    fn is_frozen(&self) -> bool { self.frozen }
    fn freeze(&mut self) { self.frozen = true; }
    fn unfreeze(&mut self) { self.frozen = false; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_shape() {
        let rn = RMSNorm::new(64, 1e-5);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = rn.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_rms_norm_with_offset_shape() {
        let rn = RMSNorm::from_weight_with_offset(Tensor::zeros(vec![64]), 1e-5, 1.0);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = rn.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_rms_norm_zero_offset_matches_standard() {
        let weight = Tensor::randn(vec![32]);
        let standard = RMSNorm::from_weight(weight.clone(), 1e-5);
        let with_offset = RMSNorm::from_weight_with_offset(weight, 1e-5, 0.0);
        let x = Tensor::randn(vec![1, 4, 32]);
        let y_std = standard.forward(&x).unwrap();
        let y_off = with_offset.forward(&x).unwrap();
        let d_std = y_std.as_slice_f32().unwrap();
        let d_off = y_off.as_slice_f32().unwrap();
        for i in 0..d_std.len() {
            assert!((d_std[i] - d_off[i]).abs() < 1e-5,
                "mismatch at index {}: {} vs {}", i, d_std[i], d_off[i]);
        }
    }

    #[test]
    fn test_rms_norm_freezable() {
        let mut rn = RMSNorm::new(64, 1e-5);
        assert!(!rn.is_frozen());
        rn.freeze();
        assert!(rn.is_frozen());
        let (total, frozen) = rn.parameter_count();
        assert_eq!(total, 64);
        assert_eq!(frozen, 64);
        rn.unfreeze();
        assert!(!rn.is_frozen());
    }

    // ==================== forward_inplace tests ====================

    #[test]
    fn test_rms_norm_inplace_matches_forward() {
        let rn = RMSNorm::new(64, 1e-5);
        let x = Tensor::randn(vec![2, 10, 64]);

        let y_alloc = rn.forward(&x).unwrap();

        let mut y_inplace = x.clone();
        rn.forward_inplace(&mut y_inplace).unwrap();

        let d_alloc = y_alloc.as_slice_f32().unwrap();
        let d_inplace = y_inplace.as_slice_f32().unwrap();
        assert_eq!(d_alloc.len(), d_inplace.len());
        for i in 0..d_alloc.len() {
            assert!((d_alloc[i] - d_inplace[i]).abs() < 1e-5,
                "mismatch at index {}: alloc={}, inplace={}", i, d_alloc[i], d_inplace[i]);
        }
    }

    #[test]
    fn test_rms_norm_inplace_shape_preserved() {
        let rn = RMSNorm::new(32, 1e-6);
        let mut x = Tensor::randn(vec![1, 4, 32]);
        rn.forward_inplace(&mut x).unwrap();
        assert_eq!(x.shape(), &[1, 4, 32]);
    }

    #[test]
    fn test_rms_norm_inplace_with_offset_matches_forward() {
        let rn = RMSNorm::from_weight_with_offset(Tensor::randn(vec![32]), 1e-5, 1.0);
        let x = Tensor::randn(vec![1, 4, 32]);

        let y_alloc = rn.forward(&x).unwrap();

        let mut y_inplace = x.clone();
        rn.forward_inplace(&mut y_inplace).unwrap();

        let d_alloc = y_alloc.as_slice_f32().unwrap();
        let d_inplace = y_inplace.as_slice_f32().unwrap();
        assert_eq!(d_alloc.len(), d_inplace.len());
        for i in 0..d_alloc.len() {
            assert!((d_alloc[i] - d_inplace[i]).abs() < 1e-5,
                "mismatch at index {}: alloc={}, inplace={}", i, d_alloc[i], d_inplace[i]);
        }
    }

    #[test]
    fn test_rms_norm_inplace_does_not_modify_original() {
        let rn = RMSNorm::new(16, 1e-5);
        let x = Tensor::randn(vec![1, 2, 16]);
        let x_data_before = x.as_slice_f32().unwrap().to_vec();

        let mut y = x.clone();
        rn.forward_inplace(&mut y).unwrap();

        // Original should be unchanged (clone should not have aliased)
        let x_data_after = x.as_slice_f32().unwrap();
        for i in 0..x_data_before.len() {
            assert_eq!(x_data_before[i], x_data_after[i],
                "original modified at index {}", i);
        }
    }
}
