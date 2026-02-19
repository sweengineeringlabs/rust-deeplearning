use crate::api::error::SwetsResult;
use crate::api::loss::Loss;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;

/// FR-604: Cross-Entropy Loss (numerically stable log-softmax + NLL).
///
/// Predictions are raw logits of shape [batch, classes].
/// Targets are one-hot encoded of shape [batch, classes].
///
/// Numerically stable log-softmax:
///   log_softmax = x - log(sum(exp(x - max(x))))
///
/// Loss = -mean(sum(targets * log_softmax, dim=-1))
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> SwetsResult<Tensor> {
        let pred_shape = predictions.shape().to_vec();
        let tgt_shape = targets.shape().to_vec();

        assert_eq!(
            pred_shape.len(),
            2,
            "CrossEntropyLoss expects 2D predictions [batch, classes], got shape {:?}",
            pred_shape
        );
        assert_eq!(
            pred_shape, tgt_shape,
            "CrossEntropyLoss: predictions shape {:?} != targets shape {:?}",
            pred_shape, tgt_shape
        );

        let batch_size = pred_shape[0];
        let num_classes = pred_shape[1];

        let pred_data = predictions.to_vec();
        let tgt_data = targets.to_vec();

        // Compute numerically stable log-softmax and NLL per sample
        let mut total_loss = 0.0f32;

        for b in 0..batch_size {
            let offset = b * num_classes;
            let row = &pred_data[offset..offset + num_classes];

            // max(x) for numerical stability
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // log(sum(exp(x - max)))
            let log_sum_exp: f32 = row
                .iter()
                .map(|&x| (x - max_val).exp())
                .sum::<f32>()
                .ln()
                + max_val;

            // log_softmax_i = x_i - log_sum_exp
            // loss for this sample = -sum(target_i * log_softmax_i)
            let mut sample_loss = 0.0f32;
            for c in 0..num_classes {
                let log_softmax = row[c] - log_sum_exp;
                sample_loss -= tgt_data[offset + c] * log_softmax;
            }

            total_loss += sample_loss;
        }

        let mean_loss = total_loss / batch_size as f32;
        let output = Tensor::from_vec(vec![mean_loss], vec![1])?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(CrossEntropyBackward {
                    batch_size,
                    num_classes,
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

/// Backward: grad = (softmax - targets) / batch_size
struct CrossEntropyBackward {
    batch_size: usize,
    num_classes: usize,
}

impl BackwardOp for CrossEntropyBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let predictions = &saved[0];
        let targets = &saved[1];

        let pred_data = predictions.to_vec();
        let tgt_data = targets.to_vec();

        let batch_size = self.batch_size;
        let num_classes = self.num_classes;

        let mut grad_data = vec![0.0f32; batch_size * num_classes];

        for b in 0..batch_size {
            let offset = b * num_classes;
            let row = &pred_data[offset..offset + num_classes];

            // Compute softmax (numerically stable)
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_vals.iter().sum();

            for c in 0..num_classes {
                let softmax_val = exp_vals[c] / sum_exp;
                // grad = (softmax - target) / batch_size
                grad_data[offset + c] = (softmax_val - tgt_data[offset + c]) / batch_size as f32;
            }
        }

        let grad_pred = Tensor::from_vec(grad_data, predictions.shape().to_vec())
            .expect("cross_entropy backward grad tensor");

        // Scale by upstream gradient
        let grad_val = grad_output.to_vec()[0];
        let grad_pred = grad_pred.mul_scalar_raw(grad_val);

        vec![grad_pred]
    }

    fn name(&self) -> &str {
        "CrossEntropyBackward"
    }
}
