use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;

/// Backward for Softmax: grad_input = softmax * (grad - sum(grad * softmax, dim))  (FR-216)
/// saved[0] = softmax_output
///
/// Implemented element-wise via to_vec() since we lack complex dim-wise broadcast ops.
/// Assumes a 2-D tensor [batch, classes] with dim = -1 or 1.
pub struct SoftmaxBackward {
    pub dim: i64,
}

impl BackwardOp for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let softmax_out = &saved[0];
        let shape = softmax_out.shape().to_vec();
        let ndim = shape.len();

        // Normalise dim to positive index
        let dim = if self.dim < 0 {
            (ndim as i64 + self.dim) as usize
        } else {
            self.dim as usize
        };

        let s_data = softmax_out.to_vec();
        let g_data = grad_output.to_vec();
        let total = s_data.len();

        // Compute the stride for the softmax dimension: product of all dims after `dim`.
        let inner_size: usize = shape[(dim + 1)..].iter().product();
        let dim_size = shape[dim];
        let outer_size: usize = total / (dim_size * inner_size);

        let mut grad_input = vec![0.0f32; total];

        // For every slice along the softmax dimension, compute:
        //   dot = sum_j (grad[j] * softmax[j])
        //   grad_input[j] = softmax[j] * (grad[j] - dot)
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // First pass: compute dot product along dim
                let mut dot = 0.0f32;
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    dot += g_data[idx] * s_data[idx];
                }
                // Second pass: compute grad_input
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    grad_input[idx] = s_data[idx] * (g_data[idx] - dot);
                }
            }
        }

        let result =
            Tensor::from_vec(grad_input, shape).expect("softmax backward from_vec");
        vec![result]
    }

    fn name(&self) -> &str {
        "SoftmaxBackward"
    }
}
