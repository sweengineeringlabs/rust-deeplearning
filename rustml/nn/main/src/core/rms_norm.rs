//! RMSNorm: x * weight / rms(x). Used by Llama-family models.
//! Unlike LayerNorm, does not subtract mean and has no bias parameter.

use crate::api::error::NnResult;
use crate::api::traits::Freezable;
use rustml_core::Tensor;

/// RMSNorm layer
#[derive(Debug, Clone)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
    pub frozen: bool,
}

impl RMSNorm {
    /// Create a new RMSNorm with weights initialized to 1.
    pub fn new(dim: usize, eps: f32) -> Self {
        let weight = Tensor::ones(vec![dim]);
        Self { weight, eps, frozen: false }
    }

    /// Create from a pre-loaded weight tensor.
    pub fn from_weight(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps, frozen: false }
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let total = self.weight.numel();
        let frozen = if self.frozen { total } else { 0 };
        (total, frozen)
    }

    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        Ok(x.rms_norm(&self.weight, self.eps)?)
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
