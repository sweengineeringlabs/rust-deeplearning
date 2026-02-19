use crate::api::error::SwetsResult;
use crate::api::optim::Optimizer;
use crate::api::tape;
use crate::api::tensor::{Tensor, TensorId};
use std::collections::HashMap;

pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocities: HashMap<TensorId, Tensor>,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            velocities: HashMap::new(),
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut Tensor]) -> SwetsResult<()> {
        for param in params.iter_mut() {
            if let Some(grad) = tape::grad(param) {
                if self.momentum > 0.0 {
                    let param_id = param.id();
                    let param_shape = param.shape().to_vec();

                    // Get or initialize velocity
                    if !self.velocities.contains_key(&param_id) {
                        self.velocities
                            .insert(param_id, Tensor::zeros(param_shape));
                    }
                    let velocity = self.velocities.get(&param_id).unwrap();

                    // v = momentum * v + grad
                    let scaled_v = velocity.mul_scalar_raw(self.momentum);
                    let new_v = scaled_v.add_raw(&grad).expect("sgd velocity add");
                    self.velocities.insert(param_id, new_v.clone());

                    // param -= lr * v
                    let update = new_v.mul_scalar_raw(self.learning_rate);
                    let new_param = param.sub_raw(&update).expect("sgd param update");
                    param.update_data_from(&new_param);
                } else {
                    // Simple SGD: param -= lr * grad
                    let update = grad.mul_scalar_raw(self.learning_rate);
                    let new_param = param.sub_raw(&update).expect("sgd param update");
                    param.update_data_from(&new_param);
                }
            }
        }
        Ok(())
    }

    fn lr(&self) -> f32 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}
