use crate::api::error::SwetsResult;
use crate::api::loss::Loss;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;

/// FR-605: Quantile Loss for quantile regression.
///
/// Asymmetric loss that penalises under-prediction and over-prediction
/// differently depending on the chosen quantile.
///
/// For each element where `error = target - prediction`:
///   if error >= 0: loss = quantile * error
///   if error <  0: loss = (quantile - 1) * error   (i.e. (1 - quantile) * |error|)
///
/// Loss = mean over all elements.
pub struct QuantileLoss {
    quantile: f32,
}

impl QuantileLoss {
    pub fn new(quantile: f32) -> Self {
        assert!(
            quantile > 0.0 && quantile < 1.0,
            "quantile must be in the open interval (0, 1), got {quantile}"
        );
        Self { quantile }
    }
}

impl Default for QuantileLoss {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Loss for QuantileLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> SwetsResult<Tensor> {
        let diff = predictions.sub_raw(targets)?;
        let diff_data = diff.to_vec();
        let n = diff_data.len();
        let q = self.quantile;

        // Element-wise quantile loss: error = target - pred = -(pred - target) = -diff
        let loss_data: Vec<f32> = diff_data
            .iter()
            .map(|&d| {
                let error = -d; // target - prediction
                if error >= 0.0 {
                    q * error
                } else {
                    (q - 1.0) * error // equals (1 - q) * |error|
                }
            })
            .collect();

        let sum: f32 = loss_data.iter().sum();
        let mean = sum / n as f32;
        let output = Tensor::from_vec(vec![mean], vec![1])?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(QuantileBackward {
                    n,
                    quantile: self.quantile,
                }),
                output_id: output.id(),
                input_ids: vec![predictions.id()],
                saved_tensors: vec![predictions.clone(), targets.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }
}

/// Backward pass for quantile loss.
///
/// For each element where `error = target - pred`:
///   if error >= 0: grad_pred = -quantile / n
///   if error <  0: grad_pred = (1 - quantile) / n
struct QuantileBackward {
    n: usize,
    quantile: f32,
}

impl BackwardOp for QuantileBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let predictions = &saved[0];
        let targets = &saved[1];

        let diff = predictions.sub_raw(targets).expect("quantile backward sub");
        let diff_data = diff.to_vec();
        let q = self.quantile;
        let n = self.n as f32;

        let grad_data: Vec<f32> = diff_data
            .iter()
            .map(|&d| {
                let error = -d; // target - prediction
                if error >= 0.0 {
                    -q / n
                } else {
                    (1.0 - q) / n
                }
            })
            .collect();

        let grad_pred = Tensor::from_vec(grad_data, diff.shape().to_vec())
            .expect("quantile backward grad tensor");

        // Scale by upstream gradient
        let grad_val = grad_output.to_vec()[0];
        let grad_pred = grad_pred.mul_scalar_raw(grad_val);

        vec![grad_pred]
    }

    fn name(&self) -> &str {
        "QuantileBackward"
    }
}
