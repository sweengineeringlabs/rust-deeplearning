use smallvec::SmallVec;

/// Shape type: stack-allocated for ≤4 dimensions (covers most tensor shapes),
/// spills to heap for higher dimensionality.
pub type Shape = SmallVec<[usize; 4]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16, // Requires half crate
    BF16,
    I8,
    U8,
    /// Block-quantized 8-bit format: 32 elements per block, 34 bytes per block
    /// (2-byte f16 scale + 32 i8 values). Not per-element sized.
    Q8_0,
    /// Block-quantized 4-bit format: 32 elements per block, 18 bytes per block
    /// (2-byte f16 scale + 16 bytes of packed 4-bit values). ~14% of F32 memory.
    Q4_0,
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
            DType::Q8_0 => 0, // Block-level sizing, not per-element
            DType::Q4_0 => 0, // Block-level sizing, not per-element
        }
    }
}
