use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;
use crate::core::ops::add::unbroadcast;

/// Linear layer: y = x @ W^T + b
/// Xavier initialization for weights, zeros for bias.
pub struct Linear {
    weight: Tensor, // [out_features, in_features]
    bias: Tensor,   // [out_features]
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier uniform initialization: scale = sqrt(6 / (fan_in + fan_out))
        let scale = (6.0 / (in_features + out_features) as f32).sqrt();
        let mut weight = Tensor::randn([out_features, in_features]);
        // Scale the random values (randn gives N(0,1), we want uniform-like spread)
        weight = weight.mul_scalar_raw(scale);
        weight.set_requires_grad(true);

        let mut bias = Tensor::zeros([out_features]);
        bias.set_requires_grad(true);

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        // y = input @ weight^T + bias
        let weight_t = self.weight.transpose_raw(-1, -2)?;
        let matmul_result = input.matmul_raw(&weight_t)?;
        let output = matmul_result.add_raw(&self.bias)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(LinearBackward {
                    input_shape: input.shape().to_vec(),
                    weight_shape: self.weight.shape().to_vec(),
                    bias_shape: self.bias.shape().to_vec(),
                }),
                output_id: output.id(),
                input_ids: vec![input.id(), self.weight.id(), self.bias.id()],
                saved_tensors: vec![input.clone(), self.weight.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

/// Combined backward for Linear: handles input, weight, and bias gradients.
/// saved[0] = input, saved[1] = weight
/// input_ids[0] = input, input_ids[1] = weight, input_ids[2] = bias
struct LinearBackward {
    input_shape: Vec<usize>,
    weight_shape: Vec<usize>,
    bias_shape: Vec<usize>,
}

impl BackwardOp for LinearBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let weight = &saved[1];

        // grad_input = grad_output @ weight  (grad_output: [B, out] @ weight: [out, in] -> [B, in])
        let grad_input = grad_output.matmul_raw(weight).expect("linear grad_input");
        let grad_input = unbroadcast(&grad_input, &self.input_shape);

        // grad_weight = grad_output^T @ input  (grad_output^T: [out, B] @ input: [B, in] -> [out, in])
        let grad_output_t = grad_output.transpose_raw(-1, -2).expect("transpose grad");
        let grad_weight = grad_output_t.matmul_raw(input).expect("linear grad_weight");
        let grad_weight = unbroadcast(&grad_weight, &self.weight_shape);

        // grad_bias = sum of grad_output over batch dimension
        let grad_bias = unbroadcast(grad_output, &self.bias_shape);

        vec![grad_input, grad_weight, grad_bias]
    }

    fn name(&self) -> &str {
        "LinearBackward"
    }
}
