use crate::api::error::SwetsResult;
use crate::api::loss::Loss;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;

/// FR-603: Huber Loss with configurable delta.
///
/// For each element:
///   if |diff| <= delta: 0.5 * diff^2
///   else:               delta * (|diff| - 0.5 * delta)
///
/// Loss = mean over all elements.
pub struct HuberLoss {
    delta: f32,
}

impl HuberLoss {
    pub fn new(delta: f32) -> Self {
        Self { delta }
    }
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Loss for HuberLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> SwetsResult<Tensor> {
        let diff = predictions.sub_raw(targets)?;
        let diff_data = diff.to_vec();
        let n = diff_data.len();
        let delta = self.delta;

        // Element-wise Huber loss
        let huber_data: Vec<f32> = diff_data
            .iter()
            .map(|&d| {
                let abs_d = d.abs();
                if abs_d <= delta {
                    0.5 * d * d
                } else {
                    delta * (abs_d - 0.5 * delta)
                }
            })
            .collect();

        let sum: f32 = huber_data.iter().sum();
        let mean = sum / n as f32;
        let output = Tensor::from_vec(vec![mean], vec![1])?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(HuberBackward {
                    n,
                    delta: self.delta,
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

/// Backward:
///   if |diff| <= delta: grad = diff / n
///   else:               grad = delta * sign(diff) / n
struct HuberBackward {
    n: usize,
    delta: f32,
}

impl BackwardOp for HuberBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let predictions = &saved[0];
        let targets = &saved[1];

        let diff = predictions.sub_raw(targets).expect("huber backward sub");
        let diff_data = diff.to_vec();
        let delta = self.delta;
        let n = self.n as f32;

        let grad_data: Vec<f32> = diff_data
            .iter()
            .map(|&d| {
                let abs_d = d.abs();
                if abs_d <= delta {
                    d / n
                } else {
                    let sign = if d > 0.0 {
                        1.0
                    } else if d < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    delta * sign / n
                }
            })
            .collect();

        let grad_pred = Tensor::from_vec(grad_data, diff.shape().to_vec())
            .expect("huber backward grad tensor");

        // Scale by upstream gradient
        let grad_val = grad_output.to_vec()[0];
        let grad_pred = grad_pred.mul_scalar_raw(grad_val);

        vec![grad_pred]
    }

    fn name(&self) -> &str {
        "HuberBackward"
    }
}
