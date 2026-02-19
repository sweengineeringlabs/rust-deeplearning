use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;

/// Backward for ReLU: grad * (input > 0)
/// saved[0] = input (pre-activation)
pub struct ReLUBackward;

impl BackwardOp for ReLUBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        // Create mask: 1.0 where input > 0, 0.0 otherwise
        let input_data = input.to_vec();
        let mask_data: Vec<f32> = input_data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
        let mask =
            Tensor::from_vec(mask_data, input.shape().to_vec()).expect("relu mask");

        let grad_input = grad_output.mul_raw(&mask).expect("relu backward mul");
        vec![grad_input]
    }

    fn name(&self) -> &str {
        "ReLUBackward"
    }
}
