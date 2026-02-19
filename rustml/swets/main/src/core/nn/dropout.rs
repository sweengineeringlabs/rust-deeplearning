use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;
use rand::Rng;

/// Dropout layer (FR-308).
///
/// During training, randomly zeroes elements with probability `p` and scales
/// the remaining elements by `1 / (1 - p)` (inverted dropout).
/// During evaluation, acts as identity.
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    /// Creates a new Dropout layer with drop probability `p`.
    ///
    /// # Panics
    /// Panics if `p` is not in [0, 1).
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {}",
            p
        );
        Self { p, training: true }
    }

    /// Switch to training mode (dropout active).
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Switch to evaluation mode (dropout inactive, identity pass-through).
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Returns whether the layer is in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Returns the drop probability.
    pub fn p(&self) -> f32 {
        self.p
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        // In eval mode or with p == 0, pass through unchanged
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let scale = 1.0 / (1.0 - self.p);

        // Generate binary mask: each element kept with probability (1 - p)
        let numel = input.numel();
        let mut rng = rand::thread_rng();
        let mask_data: Vec<f32> = (0..numel)
            .map(|_| {
                if rng.r#gen::<f32>() >= self.p {
                    scale
                } else {
                    0.0
                }
            })
            .collect();
        let mask = Tensor::from_vec(mask_data, input.shape().to_vec())?;

        // Apply mask (already scaled by 1/(1-p))
        let output = input.mul_raw(&mask)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(DropoutBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![mask],
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

/// Backward for Dropout.
///
/// The gradient flows only through the elements that were kept,
/// scaled by the same factor `1 / (1 - p)`.
/// saved[0] = mask (already incorporates the 1/(1-p) scaling).
struct DropoutBackward;

impl BackwardOp for DropoutBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let mask = &saved[0];
        // grad_input = grad_output * mask (mask already has scale baked in)
        let grad_input = grad_output.mul_raw(mask).expect("dropout backward mul");
        vec![grad_input]
    }

    fn name(&self) -> &str {
        "DropoutBackward"
    }
}
