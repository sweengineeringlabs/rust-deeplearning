use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;

/// Backward for Tanh: grad_output * (1 - tanh_output^2)  (FR-214)
/// saved[0] = tanh_output (the output of the forward pass, NOT the input)
pub struct TanhBackward;

impl BackwardOp for TanhBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let tanh_out = &saved[0];
        // tanh_output^2
        let tanh_sq = tanh_out.pow_raw(2.0);
        // 1 - tanh_output^2
        let ones = Tensor::ones(tanh_out.shape().to_vec());
        let one_minus_sq = ones.sub_raw(&tanh_sq).expect("tanh 1 - t^2");
        // grad_output * (1 - tanh_output^2)
        let grad_input = grad_output.mul_raw(&one_minus_sq).expect("tanh backward mul");
        vec![grad_input]
    }

    fn name(&self) -> &str {
        "TanhBackward"
    }
}
