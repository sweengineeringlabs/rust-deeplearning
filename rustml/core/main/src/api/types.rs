//! Core types for tensor operations

/// Device type for tensor computations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
        }
    }
}

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    #[default]
    F32,
    F16,
    BF16,
    I8,
    U8,
    /// Block-quantized 8-bit: 32 elements/block, 34 bytes/block
    Q8_0,
    /// Block-quantized 4-bit: 32 elements/block, 18 bytes/block
    Q4_0,
}

impl DType {
    /// Per-element byte size. Returns 0 for block-quantized types.
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
            DType::Q8_0 => 0,
            DType::Q4_0 => 0,
        }
    }
}
