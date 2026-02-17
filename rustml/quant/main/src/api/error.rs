use thiserror::Error;

pub type QuantResult<T> = Result<T, QuantError>;

#[derive(Debug, Error)]
pub enum QuantError {
    #[error("DType mismatch: expected {expected}, got {actual}")]
    DTypeMismatch { expected: String, actual: String },

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Block alignment error: {0}")]
    BlockAlignment(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),
}
