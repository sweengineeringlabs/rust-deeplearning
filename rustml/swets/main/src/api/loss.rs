use crate::api::error::SwetsResult;
use crate::api::tensor::Tensor;

pub trait Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> SwetsResult<Tensor>;
}
