use crate::api::error::{QuantError, QuantResult};
use crate::api::types::*;
use crate::core::simd;
use half::f16;
use rayon::prelude::*;
use std::time::Instant;

/// Quantize an f32 slice to Q8_0 format.
///
/// Each block of 32 elements is stored as:
/// - 2 bytes: f16 scale (little-endian)
/// - 32 bytes: i8 quantized values
///
/// Requires element count divisible by 32.
pub fn quantize_q8_0(data: &[f32]) -> QuantResult<Vec<u8>> {
    let n_elements = data.len();
    if n_elements % Q8_0_BLOCK_SIZE != 0 {
        return Err(QuantError::BlockAlignment(format!(
            "Q8_0 requires element count divisible by {}, got {}",
            Q8_0_BLOCK_SIZE, n_elements
        )));
    }

    let n_blocks = n_elements / Q8_0_BLOCK_SIZE;
    let mut output = vec![0u8; n_blocks * Q8_0_BLOCK_BYTES];

    for block_idx in 0..n_blocks {
        let src_offset = block_idx * Q8_0_BLOCK_SIZE;
        let dst_offset = block_idx * Q8_0_BLOCK_BYTES;
        let block = &data[src_offset..src_offset + Q8_0_BLOCK_SIZE];

        let amax = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        let scale_f16 = f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        output[dst_offset] = scale_bytes[0];
        output[dst_offset + 1] = scale_bytes[1];

        for i in 0..Q8_0_BLOCK_SIZE {
            let quantized = (block[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            output[dst_offset + 2 + i] = quantized as u8;
        }
    }

    Ok(output)
}

/// Dequantize Q8_0 bytes back to f32.
pub fn dequantize_q8_0(raw: &[u8], n_elements: usize) -> QuantResult<Vec<f32>> {
    let n_blocks = n_elements / Q8_0_BLOCK_SIZE;
    if raw.len() != n_blocks * Q8_0_BLOCK_BYTES {
        return Err(QuantError::ShapeMismatch {
            expected: vec![n_blocks * Q8_0_BLOCK_BYTES],
            actual: vec![raw.len()],
        });
    }

    let mut out_f32 = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let offset = block_idx * Q8_0_BLOCK_BYTES;
        let scale_f16 = f16::from_le_bytes([raw[offset], raw[offset + 1]]);
        let scale = scale_f16.to_f32();

        for i in 0..Q8_0_BLOCK_SIZE {
            let quantized = raw[offset + 2 + i] as i8;
            out_f32.push(quantized as f32 * scale);
        }
    }

    Ok(out_f32)
}

/// Quantize an f32 slice to Q4_0 format.
///
/// Each block of 32 elements is stored as:
/// - 2 bytes: f16 scale (little-endian)
/// - 16 bytes: 32 x 4-bit packed values (two per byte, low nibble = even index, high nibble = odd)
///
/// Values are mapped to range [0, 15] representing [-8, 7] (subtract 8 to dequant).
/// Requires element count divisible by 32.
pub fn quantize_q4_0(data: &[f32]) -> QuantResult<Vec<u8>> {
    let n_elements = data.len();
    if n_elements % Q4_0_BLOCK_SIZE != 0 {
        return Err(QuantError::BlockAlignment(format!(
            "Q4_0 requires element count divisible by {}, got {}",
            Q4_0_BLOCK_SIZE, n_elements
        )));
    }

    let n_blocks = n_elements / Q4_0_BLOCK_SIZE;
    let mut output = vec![0u8; n_blocks * Q4_0_BLOCK_BYTES];

    for block_idx in 0..n_blocks {
        let src_offset = block_idx * Q4_0_BLOCK_SIZE;
        let dst_offset = block_idx * Q4_0_BLOCK_BYTES;
        let block = &data[src_offset..src_offset + Q4_0_BLOCK_SIZE];

        let amax = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = if amax == 0.0 { 0.0 } else { amax / 8.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        let scale_f16 = f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        output[dst_offset] = scale_bytes[0];
        output[dst_offset + 1] = scale_bytes[1];

        for i in 0..16 {
            let even_idx = i * 2;
            let odd_idx = i * 2 + 1;

            let even_q = (block[even_idx] * inv_scale).round().clamp(-8.0, 7.0) as i8;
            let odd_q = (block[odd_idx] * inv_scale).round().clamp(-8.0, 7.0) as i8;

            let even_u = (even_q + 8) as u8;
            let odd_u = (odd_q + 8) as u8;

            output[dst_offset + 2 + i] = (odd_u << 4) | (even_u & 0x0F);
        }
    }

    Ok(output)
}

/// Dequantize Q4_0 bytes back to f32.
pub fn dequantize_q4_0(raw: &[u8], n_elements: usize) -> QuantResult<Vec<f32>> {
    let n_blocks = n_elements / Q4_0_BLOCK_SIZE;
    if raw.len() != n_blocks * Q4_0_BLOCK_BYTES {
        return Err(QuantError::ShapeMismatch {
            expected: vec![n_blocks * Q4_0_BLOCK_BYTES],
            actual: vec![raw.len()],
        });
    }

    let mut out_f32 = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let offset = block_idx * Q4_0_BLOCK_BYTES;
        let scale_f16 = f16::from_le_bytes([raw[offset], raw[offset + 1]]);
        let scale = scale_f16.to_f32();

        for i in 0..16 {
            let packed = raw[offset + 2 + i];
            let even_u = packed & 0x0F;
            let odd_u = (packed >> 4) & 0x0F;

            let even_val = (even_u as i8 - 8) as f32 * scale;
            let odd_val = (odd_u as i8 - 8) as f32 * scale;

            out_f32.push(even_val);
            out_f32.push(odd_val);
        }
    }

    Ok(out_f32)
}

/// Fast per-row F32 -> Q8_0 quantization for the activation hot path.
///
/// Quantizes `input` (length must be a multiple of 32) into Q8_0 blocks written
/// into pre-allocated `output` buffer (34 bytes per block).
/// Also writes the f32 scale (f16-roundtripped) into `scales`.
///
/// No allocation -- caller provides buffers.
pub fn quantize_row_q8_0(input: &[f32], output: &mut [u8], scales: &mut [f32]) {
    let n_elements = input.len();
    debug_assert!(n_elements % Q8_0_BLOCK_SIZE == 0);
    let n_blocks = n_elements / Q8_0_BLOCK_SIZE;
    debug_assert!(output.len() >= n_blocks * Q8_0_BLOCK_BYTES);
    debug_assert!(scales.len() >= n_blocks);

    for block_idx in 0..n_blocks {
        let src_offset = block_idx * Q8_0_BLOCK_SIZE;
        let dst_offset = block_idx * Q8_0_BLOCK_BYTES;
        let block = &input[src_offset..src_offset + Q8_0_BLOCK_SIZE];

        let amax = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let scale_f16 = f16::from_f32(scale);
        let scale_rt = scale_f16.to_f32();
        let inv_scale = if scale_rt == 0.0 { 0.0 } else { 1.0 / scale_rt };

        let scale_bytes = scale_f16.to_le_bytes();
        output[dst_offset] = scale_bytes[0];
        output[dst_offset + 1] = scale_bytes[1];

        for i in 0..Q8_0_BLOCK_SIZE {
            let quantized = (block[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            output[dst_offset + 2 + i] = quantized as u8;
        }

        scales[block_idx] = scale_rt;
    }
}

/// Compute F32 input x Q8_0 weights -> F32 output.
///
/// `input_data` shape: [M, in_features] (flattened f32 data)
/// `weight_bytes`: raw Q8_0 bytes for [out_features, in_features]
///
/// Returns flattened f32 output of shape [M, out_features].
pub fn matmul_f32_q8(
    input_data: &[f32],
    weight_bytes: &[u8],
    m: usize,
    in_features: usize,
    out_features: usize,
) -> QuantResult<Vec<f32>> {
    let _t = if log::log_enabled!(log::Level::Trace) { Some(Instant::now()) } else { None };
    if in_features % Q8_0_BLOCK_SIZE != 0 {
        return Err(QuantError::BlockAlignment(format!(
            "in_features {} not divisible by Q8_0 block size {}",
            in_features, Q8_0_BLOCK_SIZE
        )));
    }

    let blocks_per_row = in_features / Q8_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;

    let mut output = vec![0.0f32; m * out_features];

    if m <= 4 {
        // Small-M path: parallelize over output columns (N dimension)
        let col_chunk = (out_features / rayon::current_num_threads()).max(TILE_N);
        for row_idx in 0..m {
            let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];
            let out_row = &mut output[row_idx * out_features..(row_idx + 1) * out_features];

            out_row.par_chunks_mut(col_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let col_start = chunk_idx * col_chunk;
                    for (local_c, out_val) in out_chunk.iter_mut().enumerate() {
                        let c = col_start + local_c;
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
                            let q_i8: &[i8] = unsafe {
                                std::slice::from_raw_parts(q_bytes.as_ptr() as *const i8, Q8_0_BLOCK_SIZE)
                            };
                            dot += simd::dot_q8_block(
                                &input_row[input_offset..input_offset + Q8_0_BLOCK_SIZE],
                                q_i8,
                                scale,
                            );
                        }

                        *out_val = dot;
                    }
                });
        }
    } else {
        // Large-M path: parallelize over output rows
        output
            .par_chunks_mut(out_features)
            .enumerate()
            .for_each(|(row_idx, out_row)| {
                let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];

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
    }

    if let Some(t) = _t {
        log::trace!("[perf] quant::matmul_f32_q8 [{}x{}]x[{}x{}] {:.3}ms",
            m, in_features, out_features, in_features, t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(output)
}

/// Compute F32 input x Q4_0 weights -> F32 output (float dequant path).
///
/// Returns flattened f32 output of shape [M, out_features].
pub fn matmul_f32_q4(
    input_data: &[f32],
    weight_bytes: &[u8],
    m: usize,
    in_features: usize,
    out_features: usize,
) -> QuantResult<Vec<f32>> {
    let _t = if log::log_enabled!(log::Level::Trace) { Some(Instant::now()) } else { None };
    if in_features % Q4_0_BLOCK_SIZE != 0 {
        return Err(QuantError::BlockAlignment(format!(
            "in_features {} not divisible by Q4_0 block size {}",
            in_features, Q4_0_BLOCK_SIZE
        )));
    }

    let blocks_per_row = in_features / Q4_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_0_BLOCK_BYTES;

    let mut output = vec![0.0f32; m * out_features];

    if m <= 4 {
        // Small-M path: parallelize over output columns (N dimension)
        let col_chunk = (out_features / rayon::current_num_threads()).max(TILE_N);
        for row_idx in 0..m {
            let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];
            let out_row = &mut output[row_idx * out_features..(row_idx + 1) * out_features];

            out_row.par_chunks_mut(col_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let col_start = chunk_idx * col_chunk;
                    for (local_c, out_val) in out_chunk.iter_mut().enumerate() {
                        let c = col_start + local_c;
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

                        *out_val = dot;
                    }
                });
        }
    } else {
        // Large-M path: parallelize over output rows
        output
            .par_chunks_mut(out_features)
            .enumerate()
            .for_each(|(row_idx, out_row)| {
                let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];

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
    }

    if let Some(t) = _t {
        log::trace!("[perf] quant::matmul_f32_q4 [{}x{}]x[{}x{}] {:.3}ms",
            m, in_features, out_features, in_features, t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(output)
}

/// Compute F32 input x Q4_0 weights -> F32 output using native integer dot products.
///
/// Quantizes F32 activations to Q8_0 on-the-fly, performs i8xi8->i32 integer
/// dot products within each 32-element block, and scales once per block.
pub fn matmul_f32_q4_native(
    input_data: &[f32],
    weight_bytes: &[u8],
    m: usize,
    in_features: usize,
    out_features: usize,
) -> QuantResult<Vec<f32>> {
    let _t = if log::log_enabled!(log::Level::Trace) { Some(Instant::now()) } else { None };
    if in_features % Q4_0_BLOCK_SIZE != 0 {
        return Err(QuantError::BlockAlignment(format!(
            "in_features {} not divisible by Q4_0 block size {}",
            in_features, Q4_0_BLOCK_SIZE
        )));
    }

    let blocks_per_row = in_features / Q4_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_0_BLOCK_BYTES;

    let mut output = vec![0.0f32; m * out_features];

    if m <= 4 {
        // Small-M path: parallelize over output columns (N dimension)
        // Quantize input row once before the parallel region, then share read-only buffers.
        let col_chunk = (out_features / rayon::current_num_threads()).max(TILE_N);
        for row_idx in 0..m {
            let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];
            let out_row = &mut output[row_idx * out_features..(row_idx + 1) * out_features];

            let mut q8_buf = vec![0u8; blocks_per_row * Q8_0_BLOCK_BYTES];
            let mut q8_scales = vec![0.0f32; blocks_per_row];
            quantize_row_q8_0(input_row, &mut q8_buf, &mut q8_scales);

            out_row.par_chunks_mut(col_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let col_start = chunk_idx * col_chunk;
                    for (local_c, out_val) in out_chunk.iter_mut().enumerate() {
                        let c = col_start + local_c;
                        let w_row_offset = c * row_bytes;
                        let mut dot = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let block_offset = w_row_offset + block_idx * Q4_0_BLOCK_BYTES;
                            let w_scale_f16 = f16::from_le_bytes([
                                weight_bytes[block_offset],
                                weight_bytes[block_offset + 1],
                            ]);
                            let w_scale = w_scale_f16.to_f32();
                            let packed_q4 = &weight_bytes[block_offset + 2..block_offset + 2 + 16];

                            let q8_block_offset = block_idx * Q8_0_BLOCK_BYTES;
                            let q8_i8: &[i8] = unsafe {
                                std::slice::from_raw_parts(
                                    q8_buf[q8_block_offset + 2..q8_block_offset + 2 + Q8_0_BLOCK_SIZE]
                                        .as_ptr() as *const i8,
                                    Q8_0_BLOCK_SIZE,
                                )
                            };

                            let int_dot = simd::dot_q4q8_block(packed_q4, q8_i8);
                            dot += (int_dot as f32) * q8_scales[block_idx] * w_scale;
                        }

                        *out_val = dot;
                    }
                });
        }
    } else {
        // Large-M path: parallelize over output rows
        output
            .par_chunks_mut(out_features)
            .enumerate()
            .for_each(|(row_idx, out_row)| {
                let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];

                let mut q8_buf = vec![0u8; blocks_per_row * Q8_0_BLOCK_BYTES];
                let mut q8_scales = vec![0.0f32; blocks_per_row];
                quantize_row_q8_0(input_row, &mut q8_buf, &mut q8_scales);

                let mut col_idx = 0;
                while col_idx < out_features {
                    let tile_end = (col_idx + TILE_N).min(out_features);
                    for c in col_idx..tile_end {
                        let w_row_offset = c * row_bytes;
                        let mut dot = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let block_offset = w_row_offset + block_idx * Q4_0_BLOCK_BYTES;
                            let w_scale_f16 = f16::from_le_bytes([
                                weight_bytes[block_offset],
                                weight_bytes[block_offset + 1],
                            ]);
                            let w_scale = w_scale_f16.to_f32();
                            let packed_q4 = &weight_bytes[block_offset + 2..block_offset + 2 + 16];

                            let q8_block_offset = block_idx * Q8_0_BLOCK_BYTES;
                            let q8_i8: &[i8] = unsafe {
                                std::slice::from_raw_parts(
                                    q8_buf[q8_block_offset + 2..q8_block_offset + 2 + Q8_0_BLOCK_SIZE]
                                        .as_ptr() as *const i8,
                                    Q8_0_BLOCK_SIZE,
                                )
                            };

                            let int_dot = simd::dot_q4q8_block(packed_q4, q8_i8);
                            dot += (int_dot as f32) * q8_scales[block_idx] * w_scale;
                        }

                        out_row[c] = dot;
                    }
                    col_idx = tile_end;
                }
            });
    }

    if let Some(t) = _t {
        log::trace!("[perf] quant::matmul_f32_q4_native [{}x{}]x[{}x{}] {:.3}ms",
            m, in_features, out_features, in_features, t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(output)
}

/// Dequantize Q4_1 bytes back to f32.
///
/// Q4_1 block layout (20 bytes): [f16 d][f16 m][16 bytes packed nibbles]
/// val[i] = d * nibble[i] + m
pub fn dequantize_q4_1(raw: &[u8], n_elements: usize) -> QuantResult<Vec<f32>> {
    let n_blocks = n_elements / Q4_1_BLOCK_SIZE;
    if raw.len() != n_blocks * Q4_1_BLOCK_BYTES {
        return Err(QuantError::ShapeMismatch {
            expected: vec![n_blocks * Q4_1_BLOCK_BYTES],
            actual: vec![raw.len()],
        });
    }

    let mut out_f32 = Vec::with_capacity(n_elements);
    for block_idx in 0..n_blocks {
        let offset = block_idx * Q4_1_BLOCK_BYTES;
        let d = f16::from_le_bytes([raw[offset], raw[offset + 1]]).to_f32();
        let m = f16::from_le_bytes([raw[offset + 2], raw[offset + 3]]).to_f32();

        for i in 0..16 {
            let packed = raw[offset + 4 + i];
            let lo = (packed & 0x0F) as f32;
            let hi = ((packed >> 4) & 0x0F) as f32;

            out_f32.push(d * lo + m);
            out_f32.push(d * hi + m);
        }
    }

    Ok(out_f32)
}

/// Compute F32 input x Q4_1 weights -> F32 output (float dequant path).
pub fn matmul_f32_q4_1(
    input_data: &[f32],
    weight_bytes: &[u8],
    m: usize,
    in_features: usize,
    out_features: usize,
) -> QuantResult<Vec<f32>> {
    let _t = if log::log_enabled!(log::Level::Trace) { Some(Instant::now()) } else { None };
    if in_features % Q4_1_BLOCK_SIZE != 0 {
        return Err(QuantError::BlockAlignment(format!(
            "in_features {} not divisible by Q4_1 block size {}",
            in_features, Q4_1_BLOCK_SIZE
        )));
    }

    let blocks_per_row = in_features / Q4_1_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_1_BLOCK_BYTES;

    let mut output = vec![0.0f32; m * out_features];

    if m <= 4 {
        // Small-M path: parallelize over output columns (N dimension)
        let col_chunk = (out_features / rayon::current_num_threads()).max(TILE_N);
        for row_idx in 0..m {
            let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];
            let out_row = &mut output[row_idx * out_features..(row_idx + 1) * out_features];

            out_row.par_chunks_mut(col_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let col_start = chunk_idx * col_chunk;
                    for (local_c, out_val) in out_chunk.iter_mut().enumerate() {
                        let c = col_start + local_c;
                        let w_row_offset = c * row_bytes;
                        let mut dot = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let block_offset = w_row_offset + block_idx * Q4_1_BLOCK_BYTES;
                            let d = f16::from_le_bytes([
                                weight_bytes[block_offset],
                                weight_bytes[block_offset + 1],
                            ]).to_f32();
                            let m_val = f16::from_le_bytes([
                                weight_bytes[block_offset + 2],
                                weight_bytes[block_offset + 3],
                            ]).to_f32();
                            let input_offset = block_idx * Q4_1_BLOCK_SIZE;
                            let packed = &weight_bytes[block_offset + 4..block_offset + 4 + 16];

                            let mut block_dot = 0.0f32;
                            let mut input_sum = 0.0f32;
                            for j in 0..16 {
                                let lo = (packed[j] & 0x0F) as f32;
                                let hi = ((packed[j] >> 4) & 0x0F) as f32;
                                block_dot += input_row[input_offset + j] * lo;
                                block_dot += input_row[input_offset + j + 16] * hi;
                                input_sum += input_row[input_offset + j];
                                input_sum += input_row[input_offset + j + 16];
                            }
                            dot += d * block_dot + m_val * input_sum;
                        }

                        *out_val = dot;
                    }
                });
        }
    } else {
        // Large-M path: parallelize over output rows
        output
            .par_chunks_mut(out_features)
            .enumerate()
            .for_each(|(row_idx, out_row)| {
                let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];

                let mut col_idx = 0;
                while col_idx < out_features {
                    let tile_end = (col_idx + TILE_N).min(out_features);
                    for c in col_idx..tile_end {
                        let w_row_offset = c * row_bytes;
                        let mut dot = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let block_offset = w_row_offset + block_idx * Q4_1_BLOCK_BYTES;
                            let d = f16::from_le_bytes([
                                weight_bytes[block_offset],
                                weight_bytes[block_offset + 1],
                            ]).to_f32();
                            let m_val = f16::from_le_bytes([
                                weight_bytes[block_offset + 2],
                                weight_bytes[block_offset + 3],
                            ]).to_f32();
                            let input_offset = block_idx * Q4_1_BLOCK_SIZE;
                            let packed = &weight_bytes[block_offset + 4..block_offset + 4 + 16];

                            let mut block_dot = 0.0f32;
                            let mut input_sum = 0.0f32;
                            for j in 0..16 {
                                let lo = (packed[j] & 0x0F) as f32;
                                let hi = ((packed[j] >> 4) & 0x0F) as f32;
                                block_dot += input_row[input_offset + j] * lo;
                                block_dot += input_row[input_offset + j + 16] * hi;
                                input_sum += input_row[input_offset + j];
                                input_sum += input_row[input_offset + j + 16];
                            }
                            dot += d * block_dot + m_val * input_sum;
                        }

                        out_row[c] = dot;
                    }
                    col_idx = tile_end;
                }
            });
    }

    if let Some(t) = _t {
        log::trace!("[perf] quant::matmul_f32_q4_1 [{}x{}]x[{}x{}] {:.3}ms",
            m, in_features, out_features, in_features, t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(output)
}

/// Compute F32 input x Q4_1 weights -> F32 output using native integer dot products.
///
/// Quantizes F32 activations to Q8_0 on-the-fly, performs integer dot products
/// within each 32-element block, and applies Q4_1 scale + min per block.
pub fn matmul_f32_q4_1_native(
    input_data: &[f32],
    weight_bytes: &[u8],
    m: usize,
    in_features: usize,
    out_features: usize,
) -> QuantResult<Vec<f32>> {
    let _t = if log::log_enabled!(log::Level::Trace) { Some(Instant::now()) } else { None };
    if in_features % Q4_1_BLOCK_SIZE != 0 {
        return Err(QuantError::BlockAlignment(format!(
            "in_features {} not divisible by Q4_1 block size {}",
            in_features, Q4_1_BLOCK_SIZE
        )));
    }

    let blocks_per_row = in_features / Q4_1_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_1_BLOCK_BYTES;

    let mut output = vec![0.0f32; m * out_features];

    if m <= 4 {
        // Small-M path: parallelize over output columns (N dimension)
        // Quantize input row once before the parallel region, then share read-only buffers.
        let col_chunk = (out_features / rayon::current_num_threads()).max(TILE_N);
        for row_idx in 0..m {
            let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];
            let out_row = &mut output[row_idx * out_features..(row_idx + 1) * out_features];

            let mut q8_buf = vec![0u8; blocks_per_row * Q8_0_BLOCK_BYTES];
            let mut q8_scales = vec![0.0f32; blocks_per_row];
            quantize_row_q8_0(input_row, &mut q8_buf, &mut q8_scales);

            out_row.par_chunks_mut(col_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let col_start = chunk_idx * col_chunk;
                    for (local_c, out_val) in out_chunk.iter_mut().enumerate() {
                        let c = col_start + local_c;
                        let w_row_offset = c * row_bytes;
                        let mut dot = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let block_offset = w_row_offset + block_idx * Q4_1_BLOCK_BYTES;
                            let d = f16::from_le_bytes([
                                weight_bytes[block_offset],
                                weight_bytes[block_offset + 1],
                            ]).to_f32();
                            let m_val = f16::from_le_bytes([
                                weight_bytes[block_offset + 2],
                                weight_bytes[block_offset + 3],
                            ]).to_f32();
                            let packed_q4 = &weight_bytes[block_offset + 4..block_offset + 4 + 16];

                            let q8_block_offset = block_idx * Q8_0_BLOCK_BYTES;
                            let q8_i8: &[i8] = unsafe {
                                std::slice::from_raw_parts(
                                    q8_buf[q8_block_offset + 2..q8_block_offset + 2 + Q8_0_BLOCK_SIZE]
                                        .as_ptr() as *const i8,
                                    Q8_0_BLOCK_SIZE,
                                )
                            };

                            let (unsigned_dot, q8_sum) = simd::dot_q4_1_q8_block(packed_q4, q8_i8);
                            dot += q8_scales[block_idx] * (d * unsigned_dot as f32 + m_val * q8_sum as f32);
                        }

                        *out_val = dot;
                    }
                });
        }
    } else {
        // Large-M path: parallelize over output rows
        output
            .par_chunks_mut(out_features)
            .enumerate()
            .for_each(|(row_idx, out_row)| {
                let input_row = &input_data[row_idx * in_features..(row_idx + 1) * in_features];

                let mut q8_buf = vec![0u8; blocks_per_row * Q8_0_BLOCK_BYTES];
                let mut q8_scales = vec![0.0f32; blocks_per_row];
                quantize_row_q8_0(input_row, &mut q8_buf, &mut q8_scales);

                let mut col_idx = 0;
                while col_idx < out_features {
                    let tile_end = (col_idx + TILE_N).min(out_features);
                    for c in col_idx..tile_end {
                        let w_row_offset = c * row_bytes;
                        let mut dot = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let block_offset = w_row_offset + block_idx * Q4_1_BLOCK_BYTES;
                            let d = f16::from_le_bytes([
                                weight_bytes[block_offset],
                                weight_bytes[block_offset + 1],
                            ]).to_f32();
                            let m_val = f16::from_le_bytes([
                                weight_bytes[block_offset + 2],
                                weight_bytes[block_offset + 3],
                            ]).to_f32();
                            let packed_q4 = &weight_bytes[block_offset + 4..block_offset + 4 + 16];

                            let q8_block_offset = block_idx * Q8_0_BLOCK_BYTES;
                            let q8_i8: &[i8] = unsafe {
                                std::slice::from_raw_parts(
                                    q8_buf[q8_block_offset + 2..q8_block_offset + 2 + Q8_0_BLOCK_SIZE]
                                        .as_ptr() as *const i8,
                                    Q8_0_BLOCK_SIZE,
                                )
                            };

                            let (unsigned_dot, q8_sum) = simd::dot_q4_1_q8_block(packed_q4, q8_i8);
                            dot += q8_scales[block_idx] * (d * unsigned_dot as f32 + m_val * q8_sum as f32);
                        }

                        out_row[c] = dot;
                    }
                    col_idx = tile_end;
                }
            });
    }

    if let Some(t) = _t {
        log::trace!("[perf] quant::matmul_f32_q4_1_native [{}x{}]x[{}x{}] {:.3}ms",
            m, in_features, out_features, in_features, t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(output)
}

/// Safely convert a Vec<f32> into a Vec<u8>.
pub fn f32_vec_to_bytes(v: Vec<f32>) -> Vec<u8> {
    match bytemuck::try_cast_vec::<f32, u8>(v) {
        Ok(bytes) => bytes,
        Err((_, original)) => bytemuck::cast_slice::<f32, u8>(&original).to_vec(),
    }
}

/// Safely convert a &[f32] to &[u8].
pub fn f32_slice_to_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let quantized = quantize_q8_0(&data).unwrap();
        let dequantized = dequantize_q8_0(&quantized, 64).unwrap();

        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.05, "Q8_0 roundtrip error too large: {} vs {}", orig, deq);
        }
    }

    #[test]
    fn test_q4_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let quantized = quantize_q4_0(&data).unwrap();
        let dequantized = dequantize_q4_0(&quantized, 64).unwrap();

        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.5, "Q4_0 roundtrip error too large: {} vs {}", orig, deq);
        }
    }

    #[test]
    fn test_q8_block_alignment_error() {
        let data = vec![1.0f32; 33]; // Not divisible by 32
        assert!(quantize_q8_0(&data).is_err());
    }

    #[test]
    fn test_q4_block_alignment_error() {
        let data = vec![1.0f32; 33];
        assert!(quantize_q4_0(&data).is_err());
    }

    #[test]
    fn test_q8_matmul() {
        let in_features = 64;
        let out_features = 32;
        let m = 2;

        // Create input and weights
        let input: Vec<f32> = (0..m * in_features).map(|i| (i as f32) * 0.01).collect();
        let weight_f32: Vec<f32> = (0..out_features * in_features).map(|i| (i as f32) * 0.001).collect();
        let weight_q8 = quantize_q8_0(&weight_f32).unwrap();

        let result = matmul_f32_q8(&input, &weight_q8, m, in_features, out_features).unwrap();
        assert_eq!(result.len(), m * out_features);
    }

    #[test]
    fn test_f32_vec_to_bytes_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes = f32_vec_to_bytes(data.clone());
        assert_eq!(bytes.len(), 16);
        let back: &[f32] = bytemuck::cast_slice(&bytes);
        assert_eq!(back, &data[..]);
    }
}
