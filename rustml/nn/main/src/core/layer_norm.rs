//! Layer normalization implementation

use crate::api::error::NnResult;
use rustml_core::Tensor;

/// Layer normalization
///
/// Normalizes the input across the last dimension:
/// y = (x - mean) / sqrt(var + eps) * gamma + beta
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Learnable scale parameter (gamma)
    pub weight: Tensor,
    /// Learnable shift parameter (beta)
    pub bias: Tensor,
    /// Normalized shape (typically the last dimension)
    pub normalized_shape: usize,
    /// Small constant for numerical stability
    pub eps: f32,
}

impl LayerNorm {
    /// Create a new layer normalization
    pub fn new(normalized_shape: usize) -> Self {
        Self::with_eps(normalized_shape, 1e-5)
    }

    /// Create layer normalization with custom epsilon
    pub fn with_eps(normalized_shape: usize, eps: f32) -> Self {
        let weight = Tensor::ones(vec![normalized_shape]);
        let bias = Tensor::zeros(vec![normalized_shape]);

        Self {
            weight,
            bias,
            normalized_shape,
            eps,
        }
    }

    /// Create from existing weights
    pub fn from_weights(weight: Tensor, bias: Tensor, eps: f32) -> NnResult<Self> {
        if weight.shape() != bias.shape() {
            return Err(crate::api::error::NnError::ShapeMismatch(
                "Weight and bias shapes must match".into(),
            ));
        }
        if weight.ndim() != 1 {
            return Err(crate::api::error::NnError::InvalidConfig(
                "Weight must be 1D".into(),
            ));
        }

        Ok(Self {
            normalized_shape: weight.shape()[0],
            weight,
            bias,
            eps,
        })
    }

    /// Forward pass
    ///
    /// Input shape: [..., normalized_shape]
    /// Output shape: [..., normalized_shape]
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        let shape = x.shape();
        if shape.is_empty() || shape[shape.len() - 1] != self.normalized_shape {
            return Err(crate::api::error::NnError::ShapeMismatch(format!(
                "Expected last dimension to be {}, got {:?}",
                self.normalized_shape, shape
            )));
        }

        // Compute mean and variance along last dimension
        let mean = x.mean(-1)?;
        let var = x.var(-1)?;

        // Broadcast mean and var back to original shape
        let mean_broadcast = mean.unsqueeze(-1)?.broadcast_to(&x.shape().into())?;
        let var_broadcast = var.unsqueeze(-1)?.broadcast_to(&x.shape().into())?;

        // Normalize: (x - mean) / sqrt(var + eps)
        let x_centered = x.sub(&mean_broadcast)?;
        let std = var_broadcast.add_scalar(self.eps).sqrt();
        let normalized = x_centered.div(&std)?;

        // Scale and shift: gamma * normalized + beta
        let scaled = normalized.mul(&self.weight.broadcast_to(&x.shape().into())?)?;
        let output = scaled.add(&self.bias.broadcast_to(&x.shape().into())?)?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(64);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_layer_norm_normalization() {
        let ln = LayerNorm::new(4);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let y = ln.forward(&x).unwrap();

        // After normalization, mean should be ~0 and std should be ~1
        let mean = y.mean(-1).unwrap();
        let var = y.var(-1).unwrap();

        assert!(mean.get(&[0]).unwrap().abs() < 1e-5);
        assert!((var.get(&[0]).unwrap() - 1.0).abs() < 0.1);
    }
}
