//! Feed-forward network with support for GELU, SiLU, ReLU, and SwiGLU activations.

use crate::api::error::NnResult;
use crate::api::types::Activation;
use crate::core::linear::Linear;
use rustml_core::Tensor;

/// Feed-forward network (MLP) used in transformer blocks.
///
/// Supports standard activations (GELU, SiLU, ReLU) and gated variants (SwiGLU).
pub struct FeedForward {
    pub up_proj: Linear,
    pub down_proj: Linear,
    pub gate_proj: Option<Linear>,
    pub hidden_dim: usize,
    pub activation: Activation,
}

impl FeedForward {
    /// Create a standard feed-forward layer with GELU activation.
    pub fn new(d_model: usize, hidden_dim: usize, bias: bool) -> Self {
        Self {
            up_proj: Linear::with_bias(d_model, hidden_dim, bias),
            down_proj: Linear::with_bias(hidden_dim, d_model, bias),
            gate_proj: None,
            hidden_dim,
            activation: Activation::Gelu,
        }
    }

    /// Create a feed-forward layer with the specified activation.
    pub fn with_activation(
        d_model: usize,
        hidden_dim: usize,
        bias: bool,
        activation: Activation,
    ) -> Self {
        let gate_proj = if activation == Activation::SwiGLU {
            Some(Linear::with_bias(d_model, hidden_dim, bias))
        } else {
            None
        };
        Self {
            up_proj: Linear::with_bias(d_model, hidden_dim, bias),
            down_proj: Linear::with_bias(hidden_dim, d_model, bias),
            gate_proj,
            hidden_dim,
            activation,
        }
    }

    /// Create a SwiGLU feed-forward layer: gate(x) * silu(up(x)).
    pub fn swiglu(d_model: usize, hidden_dim: usize, bias: bool) -> Self {
        Self {
            up_proj: Linear::with_bias(d_model, hidden_dim, bias),
            down_proj: Linear::with_bias(hidden_dim, d_model, bias),
            gate_proj: Some(Linear::with_bias(d_model, hidden_dim, bias)),
            hidden_dim,
            activation: Activation::SwiGLU,
        }
    }

    /// Construct from pre-loaded projection layers (GELU default).
    pub fn from_weights(up_proj: Linear, down_proj: Linear) -> Self {
        let hidden_dim = up_proj.out_features;
        Self { up_proj, down_proj, gate_proj: None, hidden_dim, activation: Activation::Gelu }
    }

    /// Construct from pre-loaded projection layers with a specified activation.
    pub fn from_weights_with_activation(
        up_proj: Linear,
        down_proj: Linear,
        activation: Activation,
    ) -> Self {
        let hidden_dim = up_proj.out_features;
        Self { up_proj, down_proj, gate_proj: None, hidden_dim, activation }
    }

    /// Construct a SwiGLU feed-forward from pre-loaded weights.
    pub fn from_weights_swiglu(
        up_proj: Linear,
        gate_proj: Linear,
        down_proj: Linear,
    ) -> Self {
        let hidden_dim = up_proj.out_features;
        Self {
            up_proj,
            down_proj,
            gate_proj: Some(gate_proj),
            hidden_dim,
            activation: Activation::SwiGLU,
        }
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let (mut total, mut frozen) = (0, 0);
        for proj in [&self.up_proj, &self.down_proj] {
            let (t, f) = proj.parameter_count();
            total += t;
            frozen += f;
        }
        if let Some(ref gate) = self.gate_proj {
            let (t, f) = gate.parameter_count();
            total += t;
            frozen += f;
        }
        (total, frozen)
    }

    /// Toggle native Q4 integer matmul on all Linear projections.
    pub fn set_native_q4_matmul(&mut self, enabled: bool) {
        self.up_proj.set_native_q4_matmul(enabled);
        self.down_proj.set_native_q4_matmul(enabled);
        if let Some(ref mut gate) = self.gate_proj {
            gate.set_native_q4_matmul(enabled);
        }
    }

    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        match self.activation {
            Activation::SwiGLU => {
                let gate = self
                    .gate_proj
                    .as_ref()
                    .expect("SwiGLU requires gate_proj")
                    .forward(input)?;
                let gate = gate.silu();
                let up = self.up_proj.forward(input)?;
                let h = gate.mul(&up)?;
                self.down_proj.forward(&h)
            }
            _ => {
                let h = self.up_proj.forward(input)?;
                let h = match self.activation {
                    Activation::Gelu => h.gelu(),
                    Activation::Silu => h.silu(),
                    Activation::Relu => h.relu(),
                    Activation::SwiGLU => unreachable!(),
                };
                self.down_proj.forward(&h)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward_gelu() {
        let ff = FeedForward::new(64, 256, true);
        let x = Tensor::randn(vec![2, 8, 64]);
        let y = ff.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_feedforward_swiglu() {
        let ff = FeedForward::swiglu(64, 256, false);
        let x = Tensor::randn(vec![2, 8, 64]);
        let y = ff.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_feedforward_param_count() {
        // up: 64*256=16384, down: 256*64=16384, biases: 256+64=320
        let ff = FeedForward::new(64, 256, true);
        let (total, frozen) = ff.parameter_count();
        assert_eq!(total, 16384 + 256 + 16384 + 64);
        assert_eq!(frozen, 0);
    }
}
