//! Neural network traits

use crate::api::error::NnResult;
use rustml_core::Tensor;

/// Base attention trait
pub trait Attention {
    /// Compute attention output
    fn forward(&self, x: &Tensor) -> NnResult<Tensor>;
}

/// Trait for layers that support parameter freezing (for fine-tuning).
pub trait Freezable {
    fn is_frozen(&self) -> bool;
    fn freeze(&mut self);
    fn unfreeze(&mut self);
}
