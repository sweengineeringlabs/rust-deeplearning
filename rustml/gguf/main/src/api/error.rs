use thiserror::Error;

pub type GgufResult<T> = Result<T, GgufError>;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid GGUF format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported GGML type: {0}")]
    UnsupportedType(String),

    #[error("Missing metadata: {0}")]
    MissingMetadata(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
}
