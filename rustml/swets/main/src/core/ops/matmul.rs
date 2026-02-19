use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;

/// Backward for C = A @ B
/// saved[0] = A, saved[1] = B
/// grad_A = grad_output @ B^T
/// grad_B = A^T @ grad_output
pub struct MatMulBackward;

impl BackwardOp for MatMulBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let a = &saved[0];
        let b = &saved[1];

        // grad_A = grad_output @ B^T
        let b_t = b.transpose_raw(-1, -2).expect("transpose B");
        let grad_a = grad_output.matmul_raw(&b_t).expect("matmul grad_a");

        // grad_B = A^T @ grad_output
        let a_t = a.transpose_raw(-1, -2).expect("transpose A");
        let grad_b = a_t.matmul_raw(grad_output).expect("matmul grad_b");

        vec![grad_a, grad_b]
    }

    fn name(&self) -> &str {
        "MatMulBackward"
    }
}
