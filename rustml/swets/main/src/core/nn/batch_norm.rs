use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;

/// Batch Normalization for 1D inputs (FR-307).
///
/// Applies batch normalization over a 2D input `[batch, features]` or a
/// 3D input `[batch, channels, length]`:
///
///   output = gamma * (x - mean) / sqrt(var + eps) + beta
///
/// During training, batch statistics are used and running statistics are
/// updated via exponential moving average. During evaluation, the
/// accumulated running statistics are used instead.
///
/// Reference: Ioffe & Szegedy - "Batch Normalization: Accelerating Deep
/// Network Training by Reducing Internal Covariate Shift" (2015)
pub struct BatchNorm1d {
    gamma: Tensor,
    beta: Tensor,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    num_features: usize,
    eps: f32,
    momentum: f32,
    training: bool,
}

impl BatchNorm1d {
    /// Creates a new BatchNorm1d layer.
    ///
    /// `num_features` is the number of features (or channels) expected in the
    /// input. Gamma is initialized to ones, beta to zeros. Both require
    /// gradients. Running mean starts at zero, running variance at one.
    pub fn new(num_features: usize) -> Self {
        let mut gamma = Tensor::ones([num_features]);
        gamma.set_requires_grad(true);

        let mut beta = Tensor::zeros([num_features]);
        beta.set_requires_grad(true);

        Self {
            gamma,
            beta,
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            training: true,
        }
    }

    /// Creates a new BatchNorm1d layer with custom epsilon and momentum.
    pub fn with_config(num_features: usize, eps: f32, momentum: f32) -> Self {
        let mut bn = Self::new(num_features);
        bn.eps = eps;
        bn.momentum = momentum;
        bn
    }

    /// Sets the layer to training mode (uses batch statistics).
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Sets the layer to evaluation mode (uses running statistics).
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Returns whether the layer is in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Returns the epsilon value used for numerical stability.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Returns the momentum value for running statistics update.
    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Returns the number of features (channels).
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Returns a reference to the running mean.
    pub fn running_mean(&self) -> &[f32] {
        &self.running_mean
    }

    /// Returns a reference to the running variance.
    pub fn running_var(&self) -> &[f32] {
        &self.running_var
    }
}

impl Layer for BatchNorm1d {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let shape = input.shape().to_vec();
        let ndim = shape.len();
        assert!(
            ndim == 2 || ndim == 3,
            "BatchNorm1d: expected 2D [batch, features] or 3D [batch, channels, length] input, got {}D",
            ndim
        );

        let batch_size = shape[0];
        let channels = shape[1];
        assert_eq!(
            channels, self.num_features,
            "BatchNorm1d: expected {} features/channels, got {}",
            self.num_features, channels
        );

        let spatial_size = if ndim == 3 { shape[2] } else { 1 };
        // Number of elements per channel that contribute to statistics
        let count = batch_size * spatial_size;
        let count_f = count as f32;

        let data = input.to_vec();
        let gamma_data = self.gamma.to_vec();
        let beta_data = self.beta.to_vec();
        let total = data.len();

        let mut output_data = vec![0.0f32; total];
        let mut x_hat_data = vec![0.0f32; total];

        // Vectors to store per-channel mean and var+eps (needed for backward)
        let mut batch_mean = vec![0.0f32; channels];
        let mut batch_var_eps = vec![0.0f32; channels];

        if self.training {
            // Compute per-channel mean
            for c in 0..channels {
                let mut sum = 0.0f32;
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        sum += data[idx];
                    }
                }
                batch_mean[c] = sum / count_f;
            }

            // Compute per-channel variance
            let mut batch_var = vec![0.0f32; channels];
            for c in 0..channels {
                let mut sum_sq = 0.0f32;
                let mean_c = batch_mean[c];
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        let diff = data[idx] - mean_c;
                        sum_sq += diff * diff;
                    }
                }
                batch_var[c] = sum_sq / count_f;
                batch_var_eps[c] = batch_var[c] + self.eps;
            }

            // Normalize and apply affine transform
            for c in 0..channels {
                let inv_std = 1.0 / batch_var_eps[c].sqrt();
                let mean_c = batch_mean[c];
                let g = gamma_data[c];
                let b_val = beta_data[c];

                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        let norm_val = (data[idx] - mean_c) * inv_std;
                        x_hat_data[idx] = norm_val;
                        output_data[idx] = g * norm_val + b_val;
                    }
                }
            }

            // Update running statistics
            let m = self.momentum;
            for c in 0..channels {
                self.running_mean[c] = (1.0 - m) * self.running_mean[c] + m * batch_mean[c];
                self.running_var[c] = (1.0 - m) * self.running_var[c] + m * batch_var[c];
            }
        } else {
            // Eval mode: use running statistics
            for c in 0..channels {
                batch_mean[c] = self.running_mean[c];
                batch_var_eps[c] = self.running_var[c] + self.eps;
            }

            for c in 0..channels {
                let inv_std = 1.0 / batch_var_eps[c].sqrt();
                let mean_c = batch_mean[c];
                let g = gamma_data[c];
                let b_val = beta_data[c];

                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        let norm_val = (data[idx] - mean_c) * inv_std;
                        x_hat_data[idx] = norm_val;
                        output_data[idx] = g * norm_val + b_val;
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_data, shape.clone())?;
        let x_hat = Tensor::from_vec(x_hat_data, shape)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(BatchNorm1dBackward {
                    num_features: self.num_features,
                    batch_var_eps,
                    batch_size,
                    spatial_size,
                }),
                output_id: output.id(),
                input_ids: vec![input.id(), self.gamma.id(), self.beta.id()],
                saved_tensors: vec![input.clone(), x_hat, self.gamma.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }
}

/// Backward pass for BatchNorm1d.
///
/// saved[0] = input, saved[1] = x_hat (normalized), saved[2] = gamma
/// input_ids[0] = input, input_ids[1] = gamma, input_ids[2] = beta
///
/// Gradients:
/// - grad_beta = sum of grad_output over batch and spatial dimensions
/// - grad_gamma = sum of (grad_output * x_hat) over batch and spatial dimensions
/// - grad_input uses the standard batchnorm backward formula:
///     dx_hat = grad_output * gamma
///     grad_input = (1 / (N * inv_std)) * (N * dx_hat
///                  - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
///   where N = batch_size * spatial_size
struct BatchNorm1dBackward {
    num_features: usize,
    batch_var_eps: Vec<f32>,
    batch_size: usize,
    spatial_size: usize,
}

impl BackwardOp for BatchNorm1dBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let x_hat = &saved[1];
        let gamma = &saved[2];

        let grad_data = grad_output.to_vec();
        let x_hat_data = x_hat.to_vec();
        let gamma_data = gamma.to_vec();

        let channels = self.num_features;
        let batch_size = self.batch_size;
        let spatial_size = self.spatial_size;
        let count = batch_size * spatial_size;
        let count_f = count as f32;

        let total = grad_data.len();
        let mut grad_input_data = vec![0.0f32; total];
        let mut grad_gamma_data = vec![0.0f32; channels];
        let mut grad_beta_data = vec![0.0f32; channels];

        // Accumulate grad_gamma and grad_beta across batch and spatial dims
        for c in 0..channels {
            for b in 0..batch_size {
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    grad_beta_data[c] += grad_data[idx];
                    grad_gamma_data[c] += grad_data[idx] * x_hat_data[idx];
                }
            }
        }

        // Compute grad_input per channel using the batchnorm backward formula
        for c in 0..channels {
            let inv_std = 1.0 / self.batch_var_eps[c].sqrt();
            let g = gamma_data[c];

            // dx_hat = grad_output * gamma (per element in this channel)
            // sum_dx_hat = sum over batch and spatial of dx_hat
            // sum_dx_hat_x_hat = sum over batch and spatial of (dx_hat * x_hat)
            let mut sum_dx_hat = 0.0f32;
            let mut sum_dx_hat_x_hat = 0.0f32;

            for b in 0..batch_size {
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    let dx_hat = grad_data[idx] * g;
                    sum_dx_hat += dx_hat;
                    sum_dx_hat_x_hat += dx_hat * x_hat_data[idx];
                }
            }

            // grad_input = inv_std / N * (N * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat)
            for b in 0..batch_size {
                for s in 0..spatial_size {
                    let idx = b * channels * spatial_size + c * spatial_size + s;
                    let dx_hat = grad_data[idx] * g;
                    grad_input_data[idx] = inv_std / count_f
                        * (count_f * dx_hat
                            - sum_dx_hat
                            - x_hat_data[idx] * sum_dx_hat_x_hat);
                }
            }
        }

        let grad_input = Tensor::from_vec(grad_input_data, input.shape().to_vec())
            .expect("batch_norm grad_input");
        let grad_gamma = Tensor::from_vec(grad_gamma_data, vec![channels])
            .expect("batch_norm grad_gamma");
        let grad_beta =
            Tensor::from_vec(grad_beta_data, vec![channels]).expect("batch_norm grad_beta");

        vec![grad_input, grad_gamma, grad_beta]
    }

    fn name(&self) -> &str {
        "BatchNorm1dBackward"
    }
}
