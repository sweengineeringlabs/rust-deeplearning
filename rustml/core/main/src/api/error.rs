//! Error types for tensor operations

use thiserror::Error;

/// Result type alias for tensor operations
pub type TensorResult<T> = Result<T, TensorError>;

/// Errors that can occur during tensor operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Shape mismatch during operation
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Invalid dimension index
    #[error("Invalid dimension: {dim} for tensor with {ndim} dimensions")]
    InvalidDimension { dim: i64, ndim: usize },

    /// Invalid index access
    #[error("Index {index} out of bounds for dimension {dim} with size {size}")]
    IndexOutOfBounds { dim: usize, index: usize, size: usize },

    /// Invalid slice range
    #[error("Invalid slice range [{start}:{end}] for dimension with size {size}")]
    InvalidSliceRange { start: usize, end: usize, size: usize },

    /// Broadcasting error
    #[error("Cannot broadcast shapes {shape1:?} and {shape2:?}")]
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },

    /// Matrix multiplication dimension mismatch
    #[error("Matrix multiplication requires inner dimensions to match: {left} vs {right}")]
    MatmulDimensionMismatch { left: usize, right: usize },

    /// Empty tensor error
    #[error("Cannot perform operation on empty tensor")]
    EmptyTensor,

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Data conversion error
    #[error("Data conversion error: {0}")]
    ConversionError(String),

    /// DType mismatch
    #[error("DType mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch {
        expected: crate::api::types::DType,
        got: crate::api::types::DType,
    },

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(String),
}
