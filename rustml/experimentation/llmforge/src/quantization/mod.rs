use crate::core::tensor::{Tensor, DType};
use crate::error::{LLMForgeError, Result};
use half::f16;
use rayon::prelude::*;

/// Number of elements per Q8_0 block.
pub const Q8_0_BLOCK_SIZE: usize = 32;

/// Bytes per Q8_0 block: 2-byte f16 scale + 32 i8 values.
pub const Q8_0_BLOCK_BYTES: usize = 34;

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

    // Parallelize over output rows (input rows)
    output
        .par_chunks_mut(out_features)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];

            for col_idx in 0..out_features {
                let w_row_offset = col_idx * row_bytes;
                let mut dot = 0.0f32;

                // Per-block dot product with on-the-fly dequantization
                for block_idx in 0..blocks_per_row {
                    let block_offset = w_row_offset + block_idx * Q8_0_BLOCK_BYTES;

                    // Read f16 scale
                    let scale_f16 = f16::from_le_bytes([
                        weight_bytes[block_offset],
                        weight_bytes[block_offset + 1],
                    ]);
                    let scale = scale_f16.to_f32();

                    // Dot product: input[k] * (quantized[k] * scale)
                    let input_offset = block_idx * Q8_0_BLOCK_SIZE;
                    for i in 0..Q8_0_BLOCK_SIZE {
                        let quantized = weight_bytes[block_offset + 2 + i] as i8;
                        dot += input_row[input_offset + i] * (quantized as f32 * scale);
                    }
                }

                out_row[col_idx] = dot;
            }
        });

    // Build output shape: [..., out_features]
    let mut out_shape = input_shape[..ndim - 1].to_vec();
    out_shape.push(out_features);

    let out_bytes = crate::core::tensor::f32_vec_to_bytes(output);

    Ok(Tensor::new(out_bytes, out_shape, DType::F32))
}
