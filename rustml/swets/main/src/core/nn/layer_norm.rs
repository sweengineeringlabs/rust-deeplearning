use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;

/// Layer Normalization (FR-306).
///
/// Normalizes over the last dimension of the input, then applies an affine
/// transform: `output = gamma * normalized + beta`.
///
/// Reference: Ba, Kiros, Hinton - "Layer Normalization" (2016)
pub struct LayerNorm {
    gamma: Tensor,
    beta: Tensor,
    normalized_shape: Vec<usize>,
    eps: f32,
}

impl LayerNorm {
    /// Creates a new LayerNorm layer.
    ///
    /// `normalized_shape` specifies the shape of the dimensions to normalize over
    /// (typically the last dimension size, e.g., `vec![hidden_size]`).
    /// Gamma is initialized to ones, beta to zeros. Both require gradients.
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let total: usize = normalized_shape.iter().product();

        let mut gamma = Tensor::ones([total]);
        gamma.set_requires_grad(true);

        let mut beta = Tensor::zeros([total]);
        beta.set_requires_grad(true);

        Self {
            gamma,
            beta,
            normalized_shape,
            eps: 1e-5,
        }
    }

    /// Creates a new LayerNorm layer with a custom epsilon.
    pub fn with_eps(normalized_shape: Vec<usize>, eps: f32) -> Self {
        let mut ln = Self::new(normalized_shape);
        ln.eps = eps;
        ln
    }

    /// Returns the epsilon value used for numerical stability.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Returns the normalized shape.
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }
}

impl Layer for LayerNorm {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let shape = input.shape().to_vec();
        let ndim = shape.len();
        assert!(ndim >= 1, "LayerNorm input must have at least 1 dimension");

        let last_dim = shape[ndim - 1];
        let norm_size: usize = self.normalized_shape.iter().product();
        assert_eq!(
            last_dim, norm_size,
            "LayerNorm: last dim {} != normalized_shape product {}",
            last_dim, norm_size
        );

        let data = input.to_vec();
        let n = data.len() / last_dim; // number of "rows" to normalize
        let d = last_dim as f32;

        let mut normalized_data = vec![0.0f32; data.len()];
        let mut output_data = vec![0.0f32; data.len()];
        let gamma_data = self.gamma.to_vec();
        let beta_data = self.beta.to_vec();

        for i in 0..n {
            let start = i * last_dim;
            let end = start + last_dim;
            let row = &data[start..end];

            // Compute mean
            let mean: f32 = row.iter().sum::<f32>() / d;

            // Compute variance
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d;

            let inv_std = 1.0 / (var + self.eps).sqrt();

            // Normalize and apply affine
            for j in 0..last_dim {
                let norm_val = (row[j] - mean) * inv_std;
                normalized_data[start + j] = norm_val;
                output_data[start + j] = gamma_data[j] * norm_val + beta_data[j];
            }
        }

        let output = Tensor::from_vec(output_data, shape.clone())?;
        let normalized = Tensor::from_vec(normalized_data, shape)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(LayerNormBackward {
                    eps: self.eps,
                    last_dim,
                }),
                output_id: output.id(),
                input_ids: vec![input.id(), self.gamma.id(), self.beta.id()],
                saved_tensors: vec![input.clone(), normalized, self.gamma.clone()],
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

/// Backward for LayerNorm.
///
/// saved[0] = input, saved[1] = normalized (x_hat), saved[2] = gamma
/// input_ids[0] = input, input_ids[1] = gamma, input_ids[2] = beta
///
/// Gradients:
/// - grad_beta = sum of grad_output over all rows
/// - grad_gamma = sum of (grad_output * x_hat) over all rows
/// - grad_input uses the standard LayerNorm backward formula:
///     dx_hat = grad_output * gamma
///     dvar = sum(dx_hat * (x - mean) * -0.5 * (var + eps)^{-3/2})
///     dmean = sum(dx_hat * -inv_std) + dvar * sum(-2(x - mean)) / D
///     dx = dx_hat * inv_std + dvar * 2(x - mean)/D + dmean/D
struct LayerNormBackward {
    eps: f32,
    last_dim: usize,
}

impl BackwardOp for LayerNormBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let x_hat = &saved[1]; // normalized
        let gamma = &saved[2];

        let grad_data = grad_output.to_vec();
        let input_data = input.to_vec();
        let x_hat_data = x_hat.to_vec();
        let gamma_data = gamma.to_vec();

        let last_dim = self.last_dim;
        let d = last_dim as f32;
        let n = input_data.len() / last_dim;

        let mut grad_input_data = vec![0.0f32; input_data.len()];
        let mut grad_gamma_data = vec![0.0f32; last_dim];
        let mut grad_beta_data = vec![0.0f32; last_dim];

        // Accumulate grad_gamma and grad_beta across all rows
        for i in 0..n {
            let start = i * last_dim;
            for j in 0..last_dim {
                let idx = start + j;
                grad_gamma_data[j] += grad_data[idx] * x_hat_data[idx];
                grad_beta_data[j] += grad_data[idx];
            }
        }

        // Compute grad_input per row using the LayerNorm backward formula
        for i in 0..n {
            let start = i * last_dim;
            let end = start + last_dim;
            let row = &input_data[start..end];

            // Recompute mean and variance for this row
            let mean: f32 = row.iter().sum::<f32>() / d;
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            // dx_hat = grad_output * gamma
            let dx_hat: Vec<f32> = (0..last_dim)
                .map(|j| grad_data[start + j] * gamma_data[j])
                .collect();

            // Simplified LayerNorm backward:
            // grad_input[j] = (1/D) * inv_std * (D * dx_hat[j]
            //                  - sum(dx_hat) - x_hat[j] * sum(dx_hat * x_hat))
            let sum_dx_hat: f32 = dx_hat.iter().sum();
            let sum_dx_hat_x_hat: f32 = (0..last_dim)
                .map(|j| dx_hat[j] * x_hat_data[start + j])
                .sum();

            for j in 0..last_dim {
                grad_input_data[start + j] = inv_std / d
                    * (d * dx_hat[j]
                        - sum_dx_hat
                        - x_hat_data[start + j] * sum_dx_hat_x_hat);
            }
        }

        let grad_input = Tensor::from_vec(grad_input_data, input.shape().to_vec())
            .expect("layer_norm grad_input");
        let grad_gamma =
            Tensor::from_vec(grad_gamma_data, vec![last_dim]).expect("layer_norm grad_gamma");
        let grad_beta =
            Tensor::from_vec(grad_beta_data, vec![last_dim]).expect("layer_norm grad_beta");

        vec![grad_input, grad_gamma, grad_beta]
    }

    fn name(&self) -> &str {
        "LayerNormBackward"
    }
}
