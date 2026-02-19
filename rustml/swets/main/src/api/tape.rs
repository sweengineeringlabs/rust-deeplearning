use crate::api::tensor::{Tensor, TensorId};
use std::cell::RefCell;
use std::collections::HashMap;

// --- BackwardOp trait ---

pub trait BackwardOp: Send + Sync {
    /// Compute gradients for inputs given the gradient of the output.
    /// Returns one gradient per input_id (in the same order as input_ids in TapeEntry).
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor>;
    fn name(&self) -> &str;
}

// --- TapeEntry ---

pub struct TapeEntry {
    pub backward_op: Box<dyn BackwardOp>,
    pub output_id: TensorId,
    pub input_ids: Vec<TensorId>,
    pub saved_tensors: Vec<Tensor>,
}

// --- GradientTape ---

pub struct GradientTape {
    entries: Vec<TapeEntry>,
    grads: HashMap<TensorId, Tensor>,
    enabled: bool,
}

impl GradientTape {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            grads: HashMap::new(),
            enabled: true,
        }
    }

    pub fn record(&mut self, entry: TapeEntry) {
        if self.enabled {
            self.entries.push(entry);
        }
    }

    pub fn backward(&mut self, loss_id: TensorId, loss_shape: &[usize]) {
        // Seed gradient: ones with the shape of the loss
        let seed = Tensor::ones(loss_shape.to_vec());
        self.grads.insert(loss_id, seed);

        // Replay in reverse
        for i in (0..self.entries.len()).rev() {
            let output_id = self.entries[i].output_id;
            let grad_output = match self.grads.get(&output_id) {
                Some(g) => g.clone(),
                None => continue,
            };

            let input_grads = self.entries[i]
                .backward_op
                .backward(&grad_output, &self.entries[i].saved_tensors);

            for (j, input_id) in self.entries[i].input_ids.iter().enumerate() {
                if j < input_grads.len() {
                    let new_grad = &input_grads[j];
                    if let Some(existing) = self.grads.get(input_id) {
                        // Accumulate gradients
                        let accumulated = existing.add_raw(new_grad).expect("gradient accumulation");
                        self.grads.insert(*input_id, accumulated);
                    } else {
                        self.grads.insert(*input_id, new_grad.clone());
                    }
                }
            }
        }
    }

    pub fn grad(&self, id: TensorId) -> Option<&Tensor> {
        self.grads.get(&id)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.grads.clear();
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

// --- Thread-local API ---

thread_local! {
    static TAPE: RefCell<GradientTape> = RefCell::new(GradientTape::new());
}

pub fn record_op(entry: TapeEntry) {
    TAPE.with(|tape| tape.borrow_mut().record(entry));
}

pub fn backward(loss: &Tensor) {
    TAPE.with(|tape| {
        tape.borrow_mut().backward(loss.id(), loss.shape());
    });
}

pub fn grad(tensor: &Tensor) -> Option<Tensor> {
    TAPE.with(|tape| tape.borrow().grad(tensor.id()).cloned())
}

pub fn set_grad(tensor: &Tensor, grad: Tensor) {
    TAPE.with(|tape| {
        tape.borrow_mut().grads.insert(tensor.id(), grad);
    });
}

pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let was_enabled = TAPE.with(|tape| {
        let mut t = tape.borrow_mut();
        let prev = t.is_enabled();
        t.disable();
        prev
    });
    let result = f();
    if was_enabled {
        TAPE.with(|tape| tape.borrow_mut().enable());
    }
    result
}

pub fn clear_tape() {
    TAPE.with(|tape| tape.borrow_mut().clear());
}

pub fn is_recording() -> bool {
    TAPE.with(|tape| tape.borrow().is_enabled())
}
