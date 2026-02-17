//! Linear layer implementation with quantized weight support.

use crate::api::error::NnResult;
use crate::api::traits::Freezable;
use rustml_core::{DType, Tensor};

/// A fully connected linear layer: y = xW^T + b
///
/// Supports F32 weights as well as quantized weights (Q4_0, Q8_0).
/// Quantized weights are dequantized to F32 during forward.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    pub weight: Tensor,
    /// Optional bias vector [out_features]
    pub bias: Option<Tensor>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
    /// Whether this layer's parameters are frozen (for fine-tuning)
    pub frozen: bool,
    /// Use native Q4_0×Q8_0 integer matmul instead of dequantize-then-matmul
    pub use_native_q4: bool,
}

impl Linear {
    /// Create a new linear layer with random initialization (with bias).
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_bias(in_features, out_features, true)
    }

    /// Create a linear layer without bias
    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self {
        Self::with_bias(in_features, out_features, false)
    }

    /// Create a linear layer with or without bias.
    pub fn with_bias(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        // Xavier/Glorot initialization
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::randn(vec![out_features, in_features]).mul_scalar(scale);
        let bias = if use_bias {
            Some(Tensor::zeros(vec![out_features]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_features,
            out_features,
            frozen: false,
            use_native_q4: false,
        }
    }

    /// Create a linear layer from existing weights
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> NnResult<Self> {
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(crate::api::error::NnError::InvalidConfig(
                "Weight must be 2D".into(),
            ));
        }
        let out_features = shape[0];
        let in_features = shape[1];

        if let Some(ref b) = bias {
            if b.shape() != [out_features] {
                return Err(crate::api::error::NnError::ShapeMismatch(format!(
                    "Bias shape {:?} doesn't match out_features {}",
                    b.shape(),
                    out_features
                )));
            }
        }

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            frozen: false,
            use_native_q4: false,
        })
    }

    /// Returns true if the weight is in a quantized format (Q8_0 or Q4_0).
    pub fn is_quantized(&self) -> bool {
        matches!(self.weight.dtype(), DType::Q8_0 | DType::Q4_0)
    }

    /// Toggle native Q4_0×Q8_0 integer matmul for Q4_0 weights.
    pub fn set_native_q4_matmul(&mut self, enabled: bool) {
        self.use_native_q4 = enabled;
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let mut total = self.weight.numel();
        if let Some(ref b) = self.bias {
            total += b.numel();
        }
        let frozen = if self.frozen { total } else { 0 };
        (total, frozen)
    }

    /// Forward pass: y = xW^T + b
    ///
    /// Input shape: [..., in_features]
    /// Output shape: [..., out_features]
    ///
    /// For Q4_0 weights uses `rustml_quant::matmul_f32_q4` (or the native
    /// integer variant when `use_native_q4` is set).  Q8_0 weights use
    /// `rustml_quant::matmul_f32_q8`.  All other dtypes fall back to the
    /// standard dequantize-then-matmul path.
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        let output = if self.weight.dtype() == DType::Q4_0 {
            let x_f32 = x.to_f32()?;
            let x_data = x_f32.data()?;
            let w_bytes = self.weight.as_raw_bytes()?;
            let in_features = self.in_features;
            let out_features = self.out_features;
            let m = x_data.len() / in_features;

            let result_data = if self.use_native_q4 {
                rustml_quant::matmul_f32_q4_native(x_data, w_bytes, m, in_features, out_features)?
            } else {
                rustml_quant::matmul_f32_q4(x_data, w_bytes, m, in_features, out_features)?
            };

            // Reconstruct shape: replace last dim with out_features
            let mut out_shape: Vec<usize> = x.shape().to_vec();
            *out_shape.last_mut().unwrap() = out_features;
            Tensor::from_vec(result_data, out_shape)?
        } else if self.weight.dtype() == DType::Q8_0 {
            let x_f32 = x.to_f32()?;
            let x_data = x_f32.data()?;
            let w_bytes = self.weight.as_raw_bytes()?;
            let in_features = self.in_features;
            let out_features = self.out_features;
            let m = x_data.len() / in_features;

            let result_data = rustml_quant::matmul_f32_q8(x_data, w_bytes, m, in_features, out_features)?;

            let mut out_shape: Vec<usize> = x.shape().to_vec();
            *out_shape.last_mut().unwrap() = out_features;
            Tensor::from_vec(result_data, out_shape)?
        } else {
            let weight_t = self.weight.t()?;
            x.matmul(&weight_t)?
        };

        if let Some(ref bias) = self.bias {
            Ok(output.add(bias)?)
        } else {
            Ok(output)
        }
    }
}

impl Freezable for Linear {
    fn is_frozen(&self) -> bool { self.frozen }
    fn freeze(&mut self) { self.frozen = true; }
    fn unfreeze(&mut self) { self.frozen = false; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(4, 8);
        let x = Tensor::randn(vec![2, 3, 4]);
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 3, 8]);
    }

    #[test]
    fn test_linear_no_bias() {
        let linear = Linear::new_no_bias(4, 8);
        assert!(linear.bias.is_none());
        let x = Tensor::randn(vec![2, 4]);
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8]);
    }

    #[test]
    fn test_linear_with_bias() {
        let linear_bias = Linear::with_bias(4, 8, true);
        assert!(linear_bias.bias.is_some());
        let linear_no_bias = Linear::with_bias(4, 8, false);
        assert!(linear_no_bias.bias.is_none());
    }

    #[test]
    fn test_linear_freezable() {
        let mut linear = Linear::new(4, 8);
        assert!(!linear.is_frozen());
        linear.freeze();
        assert!(linear.is_frozen());
        let (total, frozen) = linear.parameter_count();
        assert_eq!(total, 4 * 8 + 8); // weight + bias
        assert_eq!(frozen, total);
        linear.unfreeze();
        assert!(!linear.is_frozen());
    }

    #[test]
    fn test_linear_parameter_count() {
        let linear = Linear::new(4, 8);
        let (total, frozen) = linear.parameter_count();
        assert_eq!(total, 4 * 8 + 8);
        assert_eq!(frozen, 0);

        let linear_nb = Linear::new_no_bias(4, 8);
        let (total, frozen) = linear_nb.parameter_count();
        assert_eq!(total, 4 * 8);
        assert_eq!(frozen, 0);
    }
}
