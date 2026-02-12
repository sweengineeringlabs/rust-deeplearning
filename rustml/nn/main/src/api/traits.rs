//! Neural network traits

use crate::api::error::NnResult;
use rustml_core::Tensor;

/// Base attention trait
pub trait Attention {
    /// Compute attention output
    fn forward(&self, x: &Tensor) -> NnResult<Tensor>;
}
