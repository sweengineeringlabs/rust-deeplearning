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
}
