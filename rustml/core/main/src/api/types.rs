//! Core types for tensor operations

/// Device type for tensor computations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    // Future: Cuda(usize), Metal, etc.
}

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    #[default]
    F32,
    F64,
    I32,
    I64,
}
