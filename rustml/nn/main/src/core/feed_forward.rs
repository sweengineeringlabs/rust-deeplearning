//! Feed-forward network with support for GELU, SiLU, ReLU, and SwiGLU activations.

use crate::api::error::NnResult;
use crate::api::types::Activation;
use crate::core::linear::Linear;
use std::time::Instant;
use rustml_core::{DType, Tensor};

/// Feed-forward network (MLP) used in transformer blocks.
///
/// Supports standard activations (GELU, SiLU, ReLU) and gated variants (SwiGLU).
pub struct FeedForward {
    pub up_proj: Linear,
    pub down_proj: Linear,
    pub gate_proj: Option<Linear>,
    pub hidden_dim: usize,
    pub activation: Activation,
    /// Fused gate+up projection for gated activations (SwiGLU, GeGLU).
    /// Created by `fuse_gate_up_weights()` — halves matmul dispatch overhead.
    pub fused_gate_up: Option<Linear>,
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
            fused_gate_up: None,
        }
    }

    /// Create a feed-forward layer with the specified activation.
    pub fn with_activation(
        d_model: usize,
        hidden_dim: usize,
        bias: bool,
        activation: Activation,
    ) -> Self {
        let gate_proj = if activation == Activation::SwiGLU || activation == Activation::GeGLU {
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
            fused_gate_up: None,
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
            fused_gate_up: None,
        }
    }

    /// Construct from pre-loaded projection layers (GELU default).
    pub fn from_weights(up_proj: Linear, down_proj: Linear) -> Self {
        let hidden_dim = up_proj.out_features;
        Self { up_proj, down_proj, gate_proj: None, hidden_dim, activation: Activation::Gelu, fused_gate_up: None }
    }

    /// Construct from pre-loaded projection layers with a specified activation.
    pub fn from_weights_with_activation(
        up_proj: Linear,
        down_proj: Linear,
        activation: Activation,
    ) -> Self {
        let hidden_dim = up_proj.out_features;
        Self { up_proj, down_proj, gate_proj: None, hidden_dim, activation, fused_gate_up: None }
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
            fused_gate_up: None,
        }
    }

    /// Construct a GeGLU feed-forward from pre-loaded weights.
    ///
    /// GeGLU is like SwiGLU but uses GELU instead of SiLU: `gelu(gate(x)) * up(x)`.
    pub fn from_weights_geglu(
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
            activation: Activation::GeGLU,
            fused_gate_up: None,
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

    /// Fuse gate_proj and up_proj Q8_0 weights into a single `[2*hidden_dim, in_features]` tensor.
    /// Halves matmul dispatch overhead for gated activations (SwiGLU, GeGLU).
    /// No-op if gate_proj is absent, weights are not both Q8_0, or biases are present.
    pub fn fuse_gate_up_weights(&mut self) -> bool {
        let gate = match self.gate_proj.as_ref() {
            Some(g) if g.weight.dtype() == DType::Q8_0 => g,
            _ => return false,
        };
        if self.up_proj.weight.dtype() != DType::Q8_0 { return false; }
        if gate.bias.is_some() || self.up_proj.bias.is_some() { return false; }
        if gate.in_features != self.up_proj.in_features { return false; }
        if gate.out_features != self.up_proj.out_features { return false; }

        let gate_bytes = match gate.weight.as_raw_bytes() {
            Ok(b) => b,
            Err(_) => return false,
        };
        let up_bytes = match self.up_proj.weight.as_raw_bytes() {
            Ok(b) => b,
            Err(_) => return false,
        };

        let mut fused_bytes = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        fused_bytes.extend_from_slice(gate_bytes);
        fused_bytes.extend_from_slice(up_bytes);

        let in_features = gate.in_features;
        let out_features = gate.out_features + self.up_proj.out_features;

        let fused_weight = Tensor::new(fused_bytes, vec![out_features, in_features], DType::Q8_0);
        self.fused_gate_up = Some(Linear {
            weight: fused_weight,
            bias: None,
            in_features,
            out_features,
            frozen: true,
            use_native_q4: false,
        });
        true
    }

    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        let _t = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
        let result = self.forward_inner(input);
        if let Some(t) = _t {
            log::debug!("[perf] ffn::forward {:?} {:.3}ms", input.shape(), t.elapsed().as_secs_f64() * 1000.0);
        }
        result
    }

    fn forward_inner(&self, input: &Tensor) -> NnResult<Tensor> {
        match self.activation {
            Activation::SwiGLU => {
                if let Some(ref fused) = self.fused_gate_up {
                    let fused_out = fused.forward(input)?;
                    let half = self.hidden_dim;
                    let gate = fused_out.slice(-1, 0, half)?.contiguous()?.silu();
                    let up = fused_out.slice(-1, half, 2 * half)?.contiguous()?;
                    let h = gate.mul(&up)?;
                    self.down_proj.forward(&h)
                } else {
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
            }
            Activation::GeGLU => {
                if let Some(ref fused) = self.fused_gate_up {
                    let fused_out = fused.forward(input)?;
                    let half = self.hidden_dim;
                    let gate = fused_out.slice(-1, 0, half)?.contiguous()?.gelu();
                    let up = fused_out.slice(-1, half, 2 * half)?.contiguous()?;
                    let h = gate.mul(&up)?;
                    self.down_proj.forward(&h)
                } else {
                    let gate = self
                        .gate_proj
                        .as_ref()
                        .expect("GeGLU requires gate_proj")
                        .forward(input)?;
                    let gate = gate.gelu();
                    let up = self.up_proj.forward(input)?;
                    let h = gate.mul(&up)?;
                    self.down_proj.forward(&h)
                }
            }
            _ => {
                let h = self.up_proj.forward(input)?;
                let h = match self.activation {
                    Activation::Gelu => h.gelu(),
                    Activation::Silu => h.silu(),
                    Activation::Relu => h.relu(),
                    Activation::SwiGLU | Activation::GeGLU => unreachable!(),
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
    fn test_feedforward_geglu() {
        let ff = FeedForward::with_activation(64, 256, false, Activation::GeGLU);
        let x = Tensor::randn(vec![2, 8, 64]);
        let y = ff.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_feedforward_geglu_from_weights() {
        let up = crate::core::linear::Linear::new_no_bias(64, 256);
        let gate = crate::core::linear::Linear::new_no_bias(64, 256);
        let down = crate::core::linear::Linear::new_no_bias(256, 64);
        let ff = FeedForward::from_weights_geglu(up, gate, down);
        assert_eq!(ff.activation, Activation::GeGLU);
        let x = Tensor::randn(vec![1, 4, 64]);
        let y = ff.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 4, 64]);
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
