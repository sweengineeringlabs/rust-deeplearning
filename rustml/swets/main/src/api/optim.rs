use crate::api::error::SwetsResult;
use crate::api::tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self, params: &mut [&mut Tensor]) -> SwetsResult<()>;
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}
