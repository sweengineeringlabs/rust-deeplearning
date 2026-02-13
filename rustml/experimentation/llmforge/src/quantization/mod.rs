pub mod simd;

use crate::core::tensor::{Tensor, DType};
use crate::error::{LLMForgeError, Result};
use half::f16;
use rayon::prelude::*;

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

/// Quantize an F32 tensor to Q8_0 format.
///
/// Each block of 32 elements is stored as:
/// - 2 bytes: f16 scale (little-endian)
/// - 32 bytes: i8 quantized values
///
/// Requires element count divisible by 32.
pub fn quantize_tensor(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype != DType::F32 {
        return Err(LLMForgeError::DTypeMismatch);
    }

    let data = tensor.as_slice_f32()?;
    let n_elements = data.len();

    if n_elements % Q8_0_BLOCK_SIZE != 0 {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![n_elements / Q8_0_BLOCK_SIZE * Q8_0_BLOCK_SIZE],
            actual: vec![n_elements],
        });
    }

    let n_blocks = n_elements / Q8_0_BLOCK_SIZE;
    let mut output = vec![0u8; n_blocks * Q8_0_BLOCK_BYTES];

    for block_idx in 0..n_blocks {
        let src_offset = block_idx * Q8_0_BLOCK_SIZE;
        let dst_offset = block_idx * Q8_0_BLOCK_BYTES;
        let block = &data[src_offset..src_offset + Q8_0_BLOCK_SIZE];

        // Find absmax
        let amax = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));

        // Compute scale
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        // Write f16 scale (little-endian)
        let scale_f16 = f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        output[dst_offset] = scale_bytes[0];
        output[dst_offset + 1] = scale_bytes[1];

        // Quantize values to i8
        for i in 0..Q8_0_BLOCK_SIZE {
            let quantized = (block[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            output[dst_offset + 2 + i] = quantized as u8;
        }
    }

    Ok(Tensor::new(output, tensor.shape().to_vec(), DType::Q8_0))
}

/// Dequantize a Q8_0 tensor back to F32.
pub fn dequantize_tensor(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype != DType::Q8_0 {
        return Err(LLMForgeError::DTypeMismatch);
    }

    let raw = tensor.as_raw_bytes()?;
    let n_elements: usize = tensor.shape().iter().product();
    let n_blocks = n_elements / Q8_0_BLOCK_SIZE;

    if raw.len() != n_blocks * Q8_0_BLOCK_BYTES {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![n_blocks * Q8_0_BLOCK_BYTES],
            actual: vec![raw.len()],
        });
    }

    let mut out_f32 = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let offset = block_idx * Q8_0_BLOCK_BYTES;

        // Read f16 scale
        let scale_f16 = f16::from_le_bytes([raw[offset], raw[offset + 1]]);
        let scale = scale_f16.to_f32();

        // Dequantize each i8 value
        for i in 0..Q8_0_BLOCK_SIZE {
            let quantized = raw[offset + 2 + i] as i8;
            out_f32.push(quantized as f32 * scale);
        }
    }

    let out_bytes = crate::core::tensor::f32_vec_to_bytes(out_f32);

    Ok(Tensor::new(out_bytes, tensor.shape().to_vec(), DType::F32))
}

/// Compute F32 input x Q8_0 weights -> F32 output.
///
/// `input` shape: [..., in_features] (F32)
/// `weights` shape: [out_features, in_features] (Q8_0)
///
/// Computes `input * W^T` — dot product of input rows with weight rows.
/// Dequantizes on-the-fly per block to avoid full materialization.
pub fn quantized_matmul(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    if input.dtype != DType::F32 {
        return Err(LLMForgeError::DTypeMismatch);
    }
    if weights.dtype != DType::Q8_0 {
        return Err(LLMForgeError::DTypeMismatch);
    }

    let w_shape = weights.shape();
    if w_shape.len() != 2 {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![2],
            actual: vec![w_shape.len()],
        });
    }

    let out_features = w_shape[0];
    let in_features = w_shape[1];
    let input_shape = input.shape();
    let ndim = input_shape.len();

    if input_shape[ndim - 1] != in_features {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![in_features],
            actual: vec![input_shape[ndim - 1]],
        });
    }

    if in_features % Q8_0_BLOCK_SIZE != 0 {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![in_features / Q8_0_BLOCK_SIZE * Q8_0_BLOCK_SIZE],
            actual: vec![in_features],
        });
    }

    // Collapse batch dims into M
    let m: usize = input_shape[..ndim - 1].iter().product();
    let input_data = input.as_slice_f32()?;
    let weight_bytes = weights.as_raw_bytes()?;

    let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;

    // Output: [M, out_features]
    let mut output = vec![0.0f32; m * out_features];

    // Parallelize over output rows (input rows), with cache-aware tiling
    output
        .par_chunks_mut(out_features)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];

            // Process weight rows in tiles of TILE_N for better cache locality
            let mut col_idx = 0;
            while col_idx < out_features {
                let tile_end = (col_idx + TILE_N).min(out_features);
                for c in col_idx..tile_end {
                    let w_row_offset = c * row_bytes;
                    let mut dot = 0.0f32;

                    for block_idx in 0..blocks_per_row {
                        let block_offset = w_row_offset + block_idx * Q8_0_BLOCK_BYTES;
                        let scale_f16 = f16::from_le_bytes([
                            weight_bytes[block_offset],
                            weight_bytes[block_offset + 1],
                        ]);
                        let scale = scale_f16.to_f32();
                        let input_offset = block_idx * Q8_0_BLOCK_SIZE;
                        let q_bytes = &weight_bytes[block_offset + 2..block_offset + 2 + Q8_0_BLOCK_SIZE];
                        // SAFETY: q_bytes contains i8 values stored as u8
                        let q_i8: &[i8] = unsafe {
                            std::slice::from_raw_parts(q_bytes.as_ptr() as *const i8, Q8_0_BLOCK_SIZE)
                        };
                        dot += simd::dot_q8_block(
                            &input_row[input_offset..input_offset + Q8_0_BLOCK_SIZE],
                            q_i8,
                            scale,
                        );
                    }

                    out_row[c] = dot;
                }
                col_idx = tile_end;
            }
        });

    // Build output shape: [..., out_features]
    let mut out_shape = input_shape[..ndim - 1].to_vec();
    out_shape.push(out_features);

    let out_bytes = crate::core::tensor::f32_vec_to_bytes(output);

    Ok(Tensor::new(out_bytes, out_shape, DType::F32))
}

/// Quantize an F32 tensor to Q4_0 format.
///
/// Each block of 32 elements is stored as:
/// - 2 bytes: f16 scale (little-endian)
/// - 16 bytes: 32 x 4-bit packed values (two per byte, low nibble = even index, high nibble = odd)
///
/// Values are mapped to range [0, 15] representing [-8, 7] (subtract 8 to dequant).
/// Requires element count divisible by 32.
pub fn quantize_tensor_q4(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype != DType::F32 {
        return Err(LLMForgeError::DTypeMismatch);
    }

    let data = tensor.as_slice_f32()?;
    let n_elements = data.len();

    if n_elements % Q4_0_BLOCK_SIZE != 0 {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![n_elements / Q4_0_BLOCK_SIZE * Q4_0_BLOCK_SIZE],
            actual: vec![n_elements],
        });
    }

    let n_blocks = n_elements / Q4_0_BLOCK_SIZE;
    let mut output = vec![0u8; n_blocks * Q4_0_BLOCK_BYTES];

    for block_idx in 0..n_blocks {
        let src_offset = block_idx * Q4_0_BLOCK_SIZE;
        let dst_offset = block_idx * Q4_0_BLOCK_BYTES;
        let block = &data[src_offset..src_offset + Q4_0_BLOCK_SIZE];

        // Find absmax
        let amax = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));

        // Compute scale: map range [-amax, amax] to [-8, 7]
        let scale = if amax == 0.0 { 0.0 } else { amax / 8.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        // Write f16 scale (little-endian)
        let scale_f16 = f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        output[dst_offset] = scale_bytes[0];
        output[dst_offset + 1] = scale_bytes[1];

        // Pack two 4-bit values per byte
        for i in 0..16 {
            let even_idx = i * 2;
            let odd_idx = i * 2 + 1;

            // Quantize to [-8, 7], then store as unsigned [0, 15]
            let even_q = (block[even_idx] * inv_scale).round().clamp(-8.0, 7.0) as i8;
            let odd_q = (block[odd_idx] * inv_scale).round().clamp(-8.0, 7.0) as i8;

            let even_u = (even_q + 8) as u8; // [0, 15]
            let odd_u = (odd_q + 8) as u8;   // [0, 15]

            // Low nibble = even index, high nibble = odd index
            output[dst_offset + 2 + i] = (odd_u << 4) | (even_u & 0x0F);
        }
    }

    Ok(Tensor::new(output, tensor.shape().to_vec(), DType::Q4_0))
}

/// Dequantize a Q4_0 tensor back to F32.
pub fn dequantize_tensor_q4(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype != DType::Q4_0 {
        return Err(LLMForgeError::DTypeMismatch);
    }

    let raw = tensor.as_raw_bytes()?;
    let n_elements: usize = tensor.shape().iter().product();
    let n_blocks = n_elements / Q4_0_BLOCK_SIZE;

    if raw.len() != n_blocks * Q4_0_BLOCK_BYTES {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![n_blocks * Q4_0_BLOCK_BYTES],
            actual: vec![raw.len()],
        });
    }

    let mut out_f32 = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let offset = block_idx * Q4_0_BLOCK_BYTES;

        // Read f16 scale
        let scale_f16 = f16::from_le_bytes([raw[offset], raw[offset + 1]]);
        let scale = scale_f16.to_f32();

        // Unpack 4-bit values
        for i in 0..16 {
            let packed = raw[offset + 2 + i];
            let even_u = packed & 0x0F;         // low nibble
            let odd_u = (packed >> 4) & 0x0F;   // high nibble

            let even_val = (even_u as i8 - 8) as f32 * scale;
            let odd_val = (odd_u as i8 - 8) as f32 * scale;

            out_f32.push(even_val);
            out_f32.push(odd_val);
        }
    }

    let out_bytes = crate::core::tensor::f32_vec_to_bytes(out_f32);

    Ok(Tensor::new(out_bytes, tensor.shape().to_vec(), DType::F32))
}

/// Compute F32 input x Q4_0 weights -> F32 output.
///
/// `input` shape: [..., in_features] (F32)
/// `weights` shape: [out_features, in_features] (Q4_0)
///
/// Computes `input * W^T` — dot product of input rows with weight rows.
/// Dequantizes on-the-fly per block to avoid full materialization.
pub fn quantized_matmul_q4(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    if input.dtype != DType::F32 {
        return Err(LLMForgeError::DTypeMismatch);
    }
    if weights.dtype != DType::Q4_0 {
        return Err(LLMForgeError::DTypeMismatch);
    }

    let w_shape = weights.shape();
    if w_shape.len() != 2 {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![2],
            actual: vec![w_shape.len()],
        });
    }

    let out_features = w_shape[0];
    let in_features = w_shape[1];
    let input_shape = input.shape();
    let ndim = input_shape.len();

    if input_shape[ndim - 1] != in_features {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![in_features],
            actual: vec![input_shape[ndim - 1]],
        });
    }

    if in_features % Q4_0_BLOCK_SIZE != 0 {
        return Err(LLMForgeError::ShapeMismatch {
            expected: vec![in_features / Q4_0_BLOCK_SIZE * Q4_0_BLOCK_SIZE],
            actual: vec![in_features],
        });
    }

    let m: usize = input_shape[..ndim - 1].iter().product();
    let input_data = input.as_slice_f32()?;
    let weight_bytes = weights.as_raw_bytes()?;

    let blocks_per_row = in_features / Q4_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_0_BLOCK_BYTES;

    let mut output = vec![0.0f32; m * out_features];

    output
        .par_chunks_mut(out_features)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];

            // Process weight rows in tiles of TILE_N for better cache locality
            let mut col_idx = 0;
            while col_idx < out_features {
                let tile_end = (col_idx + TILE_N).min(out_features);
                for c in col_idx..tile_end {
                    let w_row_offset = c * row_bytes;
                    let mut dot = 0.0f32;

                    for block_idx in 0..blocks_per_row {
                        let block_offset = w_row_offset + block_idx * Q4_0_BLOCK_BYTES;
                        let scale_f16 = f16::from_le_bytes([
                            weight_bytes[block_offset],
                            weight_bytes[block_offset + 1],
                        ]);
                        let scale = scale_f16.to_f32();
                        let input_offset = block_idx * Q4_0_BLOCK_SIZE;
                        let packed = &weight_bytes[block_offset + 2..block_offset + 2 + 16];
                        dot += simd::dot_q4_block(
                            &input_row[input_offset..input_offset + Q4_0_BLOCK_SIZE],
                            packed,
                            scale,
                        );
                    }

                    out_row[c] = dot;
                }
                col_idx = tile_end;
            }
        });

    let mut out_shape = input_shape[..ndim - 1].to_vec();
    out_shape.push(out_features);

    let out_bytes = crate::core::tensor::f32_vec_to_bytes(output);

    Ok(Tensor::new(out_bytes, out_shape, DType::F32))
}
