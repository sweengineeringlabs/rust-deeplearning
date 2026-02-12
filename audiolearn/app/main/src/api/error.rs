//! Error types

use std::fmt;

/// Application error type
#[derive(Debug, Clone)]
pub enum AppError {
    NotFound(String),
    Network(String),
    Audio(String),
    Storage(String),
    Auth(String),
    Tts(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(msg) => write!(f, "Not found: {}", msg),
            Self::Network(msg) => write!(f, "Network error: {}", msg),
            Self::Audio(msg) => write!(f, "Audio error: {}", msg),
            Self::Storage(msg) => write!(f, "Storage error: {}", msg),
            Self::Auth(msg) => write!(f, "Auth error: {}", msg),
            Self::Tts(msg) => write!(f, "TTS error: {}", msg),
        }
    }
}

impl std::error::Error for AppError {}

pub type Result<T> = std::result::Result<T, AppError>;

/// Alias for backward compatibility
pub type AudioLearnError = AppError;
