use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::loss::Loss;
use crate::api::optim::Optimizer;
use crate::api::tape;
use crate::api::tensor::Tensor;

pub struct Trainer<M, O, L> {
    pub model: M,
    pub optimizer: O,
    pub loss_fn: L,
}

impl<M: Layer, O: Optimizer, L: Loss> Trainer<M, O, L> {
    pub fn new(model: M, optimizer: O, loss_fn: L) -> Self {
        Self {
            model,
            optimizer,
            loss_fn,
        }
    }

    /// Train one epoch over the provided batches.
    /// Each batch is (input, target).
    /// Returns the average loss over the epoch.
    pub fn train_epoch(&mut self, batches: &[(Tensor, Tensor)]) -> SwetsResult<f32> {
        let mut total_loss = 0.0;

        for (input, target) in batches {
            tape::clear_tape();

            let output = self.model.forward(input)?;
            let loss = self.loss_fn.forward(&output, target)?;
            total_loss += loss.to_vec()[0];

            tape::backward(&loss);

            let mut params: Vec<&mut Tensor> = self.model.parameters_mut();
            let mut param_refs: Vec<&mut Tensor> =
                params.iter_mut().map(|p| &mut **p).collect();
            self.optimizer.step(&mut param_refs)?;
        }

        Ok(total_loss / batches.len() as f32)
    }

    /// Validate on the provided batches (no gradient computation).
    /// Returns the average loss.
    pub fn validate(&mut self, batches: &[(Tensor, Tensor)]) -> SwetsResult<f32> {
        let mut total_loss = 0.0;

        tape::no_grad(|| {
            for (input, target) in batches {
                let output = self.model.forward(input).expect("validate forward");
                let loss = self.loss_fn.forward(&output, target).expect("validate loss");
                total_loss += loss.to_vec()[0];
            }
        });

        Ok(total_loss / batches.len() as f32)
    }
}
