use crate::api::error::SwetsResult;
use crate::api::tensor::Tensor;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    fn parameter_count(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}
