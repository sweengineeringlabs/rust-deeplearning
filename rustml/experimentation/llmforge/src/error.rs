use thiserror::Error;

#[derive(Debug, Error)]
pub enum LLMForgeError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch { expected: String, actual: String },

    #[error("Out of memory: tried to allocate {size} bytes")]
    OutOfMemory { size: usize },

    #[error("Index out of bounds: index {index} is out of bounds for dim {dim} with size {size}")]
    IndexOutOfBounds { index: usize, dim: usize, size: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Sequence length exceeded: max {max}, actual {actual}")]
    SequenceLengthExceeded { max: usize, actual: usize },

    #[error("Unknown DType byte: {0}")]
    UnknownDType(u8),

    #[error("Unsupported DType conversion")]
    DTypeMismatch,

    #[error("Operation not implemented: {0}")]
    NotImplemented(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
}

pub type Result<T> = std::result::Result<T, LLMForgeError>;
