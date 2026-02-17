/// Number of elements per Q8_0 block.
pub const Q8_0_BLOCK_SIZE: usize = 32;

/// Bytes per Q8_0 block: 2-byte f16 scale + 32 i8 values.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Number of elements per Q4_0 block.
pub const Q4_0_BLOCK_SIZE: usize = 32;

/// Bytes per Q4_0 block: 2-byte f16 scale + 16 bytes (32 x 4-bit packed).
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Tile size for input rows (cache-aware tiled matmul).
pub const TILE_M: usize = 4;

/// Tile size for weight rows (cache-aware tiled matmul).
pub const TILE_N: usize = 8;
