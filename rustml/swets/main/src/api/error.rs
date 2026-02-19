use rustml_core::TensorError;
use thiserror::Error;

pub type SwetsResult<T> = Result<T, SwetsError>;

#[derive(Debug, Error)]
pub enum SwetsError {
    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),

    #[error("Tape error: {0}")]
    TapeError(String),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Training error: {0}")]
    TrainingError(String),
}
