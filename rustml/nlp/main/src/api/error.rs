//! Error types for NLP operations

use thiserror::Error;

/// Result type for NLP operations
pub type NlpResult<T> = Result<T, NlpError>;

/// Errors that can occur in NLP operations
#[derive(Error, Debug)]
pub enum NlpError {
    #[error("Tensor error: {0}")]
    TensorError(#[from] rustml_core::TensorError),

    #[error("Neural network error: {0}")]
    NnError(#[from] rustml_nn::NnError),

    #[error("Hub error: {0}")]
    HubError(#[from] rustml_hub::HubError),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Generation error: {0}")]
    GenerationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
