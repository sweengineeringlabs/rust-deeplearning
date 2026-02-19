use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;

/// Backward for Sigmoid: grad_output * sigmoid_output * (1 - sigmoid_output)  (FR-213)
/// saved[0] = sigmoid_output (the output of the forward pass, NOT the input)
pub struct SigmoidBackward;

impl BackwardOp for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let sig = &saved[0];
        // (1 - sigmoid_output)
        let ones = Tensor::ones(sig.shape().to_vec());
        let one_minus_sig = ones.sub_raw(sig).expect("sigmoid 1 - s");
        // sigmoid_output * (1 - sigmoid_output)
        let local_grad = sig.mul_raw(&one_minus_sig).expect("sigmoid s*(1-s)");
        // grad_output * local_grad
        let grad_input = grad_output.mul_raw(&local_grad).expect("sigmoid backward mul");
        vec![grad_input]
    }

    fn name(&self) -> &str {
        "SigmoidBackward"
    }
}
