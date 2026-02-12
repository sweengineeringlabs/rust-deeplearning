//! Linear layer implementation

use crate::api::error::NnResult;
use rustml_core::Tensor;

/// A fully connected linear layer: y = xW^T + b
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
}

impl Linear {
    /// Create a new linear layer with random initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier/Glorot initialization
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::randn(vec![out_features, in_features]).mul_scalar(scale);
        let bias = Some(Tensor::zeros(vec![out_features]));

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Create a linear layer without bias
    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self {
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::randn(vec![out_features, in_features]).mul_scalar(scale);

        Self {
            weight,
            bias: None,
            in_features,
            out_features,
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
        })
    }

    /// Forward pass: y = xW^T + b
    ///
    /// Input shape: [..., in_features]
    /// Output shape: [..., out_features]
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        // x: [..., in_features]
        // weight: [out_features, in_features]
        // We need x @ weight.T
        let weight_t = self.weight.t()?;
        let mut out = x.matmul(&weight_t)?;

        if let Some(ref bias) = self.bias {
            out = out.add(bias)?;
        }

        Ok(out)
    }
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
}
