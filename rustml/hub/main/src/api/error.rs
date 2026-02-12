//! Error types for hub operations

use thiserror::Error;

/// Result type for hub operations
pub type HubResult<T> = Result<T, HubError>;

/// Errors that can occur in hub operations
#[derive(Error, Debug)]
pub enum HubError {
    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Weight loading error: {0}")]
    WeightLoadError(String),

    #[error("SafeTensors error: {0}")]
    SafeTensorsError(#[from] crate::core::safetensors::SafeTensorsError),

    #[error("Tensor error: {0}")]
    TensorError(#[from] rustml_core::TensorError),
}
