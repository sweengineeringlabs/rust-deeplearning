use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;
use crate::core::ops::add::unbroadcast;

/// Backward for C = A * B (element-wise)
/// saved[0] = A, saved[1] = B
/// grad_A = grad_output * B
/// grad_B = grad_output * A
pub struct MulBackward;

impl BackwardOp for MulBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let a = &saved[0];
        let b = &saved[1];

        let grad_a_full = grad_output.mul_raw(b).expect("mul grad_a");
        let grad_b_full = grad_output.mul_raw(a).expect("mul grad_b");

        let grad_a = unbroadcast(&grad_a_full, a.shape());
        let grad_b = unbroadcast(&grad_b_full, b.shape());

        vec![grad_a, grad_b]
    }

    fn name(&self) -> &str {
        "MulBackward"
    }
}
