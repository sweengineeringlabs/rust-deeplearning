use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;
use crate::core::ops::relu::ReLUBackward;
use crate::core::ops::sigmoid::SigmoidBackward;
use crate::core::ops::tanh::TanhBackward;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let output = input.relu_raw();

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(ReLUBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![input.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// Sigmoid (FR-305)
// ---------------------------------------------------------------------------

pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let output = Tensor::new(input.inner().sigmoid(), false);

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(SigmoidBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![output.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// Tanh (FR-305)
// ---------------------------------------------------------------------------

pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Tanh {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let output = Tensor::new(input.inner().tanh(), false);

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(TanhBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![output.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// GELU (FR-305) - Approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ---------------------------------------------------------------------------

/// Backward op for the approximate GELU activation.
/// saved[0] = input (pre-activation)
pub struct GELUBackward;

impl BackwardOp for GELUBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let x_data = input.to_vec();
        let grad_data = grad_output.to_vec();

        let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();

        let grad_input: Vec<f32> = x_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let x3 = x * x * x;
                let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner * tanh_inner;
                let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
                // d/dx GELU = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner
                let d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;
                g * d_gelu
            })
            .collect();

        let result =
            Tensor::from_vec(grad_input, input.shape().to_vec()).expect("gelu backward from_vec");
        vec![result]
    }

    fn name(&self) -> &str {
        "GELUBackward"
    }
}

pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GELU {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let x_data = input.to_vec();
        let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();

        let output_data: Vec<f32> = x_data
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();

        let output = Tensor::from_vec(output_data, input.shape().to_vec())?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(GELUBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![input.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// SiLU (FR-305) - x * sigmoid(x)
// ---------------------------------------------------------------------------

/// Backward op for the SiLU (Swish) activation.
/// saved[0] = input (pre-activation)
/// Gradient: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
pub struct SiLUBackward;

impl BackwardOp for SiLUBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let x_data = input.to_vec();
        let grad_data = grad_output.to_vec();

        let grad_input: Vec<f32> = x_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let sig = 1.0 / (1.0 + (-x).exp());
                let d_silu = sig * (1.0 + x * (1.0 - sig));
                g * d_silu
            })
            .collect();

        let result =
            Tensor::from_vec(grad_input, input.shape().to_vec()).expect("silu backward from_vec");
        vec![result]
    }

    fn name(&self) -> &str {
        "SiLUBackward"
    }
}

pub struct SiLU;

impl SiLU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for SiLU {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let x_data = input.to_vec();

        let output_data: Vec<f32> = x_data
            .iter()
            .map(|&x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            })
            .collect();

        let output = Tensor::from_vec(output_data, input.shape().to_vec())?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(SiLUBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![input.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}
