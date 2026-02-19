use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, TapeEntry};
use crate::api::tensor::Tensor;
use crate::core::ops::relu::ReLUBackward;

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
