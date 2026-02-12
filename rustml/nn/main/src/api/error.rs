//! Error types for neural network operations

use rustml_core::TensorError;
use thiserror::Error;

/// Result type for neural network operations
pub type NnResult<T> = Result<T, NnError>;

/// Errors that can occur in neural network operations
#[derive(Error, Debug)]
pub enum NnError {
    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Weight initialization error: {0}")]
    WeightInitError(String),
}
