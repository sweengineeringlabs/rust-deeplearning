/// SIMD-accelerated dot product kernels for quantized tensor operations.
///
/// Uses `std::arch` target-specific intrinsics with scalar fallbacks.
/// No nightly features or extra crates required.
///
/// Dispatch hierarchy:
/// - x86_64: AVX2 (8 f32/cycle) → SSE2 (4 f32/cycle) → scalar
/// - aarch64: NEON (4 f32/cycle) → scalar
/// - Other: scalar fallback

/// Scalar dot product for one Q8_0 block (32 elements).
/// input: &[f32; 32], quantized: &[i8; 32], scale: f32 → f32
fn dot_q8_block_scalar(input: &[f32], quantized: &[i8], scale: f32) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..32 {
        sum += input[i] * (quantized[i] as f32);
    }
    sum * scale
}

/// Scalar dot product for one Q4_0 block (32 elements).
/// input: &[f32; 32], packed: &[u8; 16], scale: f32 → f32
/// Packed format: low nibble = even index, high nibble = odd index.
/// Values stored as unsigned 0-15, subtract 8 to dequant.
fn dot_q4_block_scalar(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..16 {
        let byte = packed[i];
        let even_val = (byte & 0x0F) as i8 - 8;
        let odd_val = ((byte >> 4) & 0x0F) as i8 - 8;
        sum += input[i * 2] * (even_val as f32);
        sum += input[i * 2 + 1] * (odd_val as f32);
    }
    sum * scale
}

// --- x86_64 SIMD implementations ---

#[cfg(target_arch = "x86_64")]
mod x86 {
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_q8_block_avx2(input: &[f32], quantized: &[i8], scale: f32) -> f32 {
        use std::arch::x86_64::*;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // Process 32 elements in 4 chunks of 8
        for chunk in 0..4 {
            let base = chunk * 8;
            let inp = _mm256_loadu_ps(input.as_ptr().add(base));

            // Convert i8 to f32: load 8 i8 values
            let mut q_f32 = [0.0f32; 8];
            for j in 0..8 {
                q_f32[j] = quantized[base + j] as f32;
            }
            let q_vec = _mm256_loadu_ps(q_f32.as_ptr());

            let prod = _mm256_mul_ps(inp, q_vec);
            match chunk {
                0 => acc0 = prod,
                1 => acc1 = prod,
                2 => acc2 = prod,
                _ => acc3 = prod,
            }
        }

        // Horizontal sum: acc0 + acc1 + acc2 + acc3
        let sum01 = _mm256_add_ps(acc0, acc1);
        let sum23 = _mm256_add_ps(acc2, acc3);
        let sum = _mm256_add_ps(sum01, sum23);

        // Reduce 8 floats to 1
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);

        _mm_cvtss_f32(result) * scale
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_q4_block_avx2(input: &[f32], packed: &[u8], scale: f32) -> f32 {
        use std::arch::x86_64::*;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // Unpack all 32 values first
        let mut unpacked = [0.0f32; 32];
        for i in 0..16 {
            let byte = packed[i];
            unpacked[i * 2] = ((byte & 0x0F) as i8 - 8) as f32;
            unpacked[i * 2 + 1] = (((byte >> 4) & 0x0F) as i8 - 8) as f32;
        }

        // Process 32 elements in 4 chunks of 8
        for chunk in 0..4 {
            let base = chunk * 8;
            let inp = _mm256_loadu_ps(input.as_ptr().add(base));
            let q_vec = _mm256_loadu_ps(unpacked.as_ptr().add(base));
            let prod = _mm256_mul_ps(inp, q_vec);
            match chunk {
                0 => acc0 = prod,
                1 => acc1 = prod,
                2 => acc2 = prod,
                _ => acc3 = prod,
            }
        }

        let sum01 = _mm256_add_ps(acc0, acc1);
        let sum23 = _mm256_add_ps(acc2, acc3);
        let sum = _mm256_add_ps(sum01, sum23);

        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);

        _mm_cvtss_f32(result) * scale
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn dot_q8_block_sse2(input: &[f32], quantized: &[i8], scale: f32) -> f32 {
        use std::arch::x86_64::*;

        let mut acc = _mm_setzero_ps();

        // Process 32 elements in 8 chunks of 4
        for chunk in 0..8 {
            let base = chunk * 4;
            let inp = _mm_loadu_ps(input.as_ptr().add(base));

            let mut q_f32 = [0.0f32; 4];
            for j in 0..4 {
                q_f32[j] = quantized[base + j] as f32;
            }
            let q_vec = _mm_loadu_ps(q_f32.as_ptr());

            let prod = _mm_mul_ps(inp, q_vec);
            acc = _mm_add_ps(acc, prod);
        }

        // Horizontal sum of 4 floats
        let shuf = _mm_movehdup_ps(acc);
        let sums = _mm_add_ps(acc, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);

        _mm_cvtss_f32(result) * scale
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn dot_q4_block_sse2(input: &[f32], packed: &[u8], scale: f32) -> f32 {
        use std::arch::x86_64::*;

        let mut acc = _mm_setzero_ps();

        // Unpack all 32 values
        let mut unpacked = [0.0f32; 32];
        for i in 0..16 {
            let byte = packed[i];
            unpacked[i * 2] = ((byte & 0x0F) as i8 - 8) as f32;
            unpacked[i * 2 + 1] = (((byte >> 4) & 0x0F) as i8 - 8) as f32;
        }

        // Process 32 elements in 8 chunks of 4
        for chunk in 0..8 {
            let base = chunk * 4;
            let inp = _mm_loadu_ps(input.as_ptr().add(base));
            let q_vec = _mm_loadu_ps(unpacked.as_ptr().add(base));
            let prod = _mm_mul_ps(inp, q_vec);
            acc = _mm_add_ps(acc, prod);
        }

        let shuf = _mm_movehdup_ps(acc);
        let sums = _mm_add_ps(acc, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);

        _mm_cvtss_f32(result) * scale
    }
}

// --- aarch64 SIMD implementations ---

#[cfg(target_arch = "aarch64")]
mod arm {
    pub(super) unsafe fn dot_q8_block_neon(input: &[f32], quantized: &[i8], scale: f32) -> f32 {
        use std::arch::aarch64::*;

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        // Process 32 elements in 8 chunks of 4
        for chunk in 0..8 {
            let base = chunk * 4;
            let inp = vld1q_f32(input.as_ptr().add(base));
            let q_f32 = [
                quantized[base] as f32,
                quantized[base + 1] as f32,
                quantized[base + 2] as f32,
                quantized[base + 3] as f32,
            ];
            let q_vec = vld1q_f32(q_f32.as_ptr());
            if chunk % 2 == 0 {
                acc0 = vfmaq_f32(acc0, inp, q_vec);
            } else {
                acc1 = vfmaq_f32(acc1, inp, q_vec);
            }
        }

        let sum = vaddq_f32(acc0, acc1);
        vaddvq_f32(sum) * scale
    }

    pub(super) unsafe fn dot_q4_block_neon(input: &[f32], packed: &[u8], scale: f32) -> f32 {
        use std::arch::aarch64::*;

        let mut unpacked = [0.0f32; 32];
        for i in 0..16 {
            let byte = packed[i];
            unpacked[i * 2] = ((byte & 0x0F) as i8 - 8) as f32;
            unpacked[i * 2 + 1] = (((byte >> 4) & 0x0F) as i8 - 8) as f32;
        }

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        for chunk in 0..8 {
            let base = chunk * 4;
            let inp = vld1q_f32(input.as_ptr().add(base));
            let q_vec = vld1q_f32(unpacked.as_ptr().add(base));
            if chunk % 2 == 0 {
                acc0 = vfmaq_f32(acc0, inp, q_vec);
            } else {
                acc1 = vfmaq_f32(acc1, inp, q_vec);
            }
        }

        let sum = vaddq_f32(acc0, acc1);
        vaddvq_f32(sum) * scale
    }
}

// --- Public dispatch functions ---

/// Runtime-dispatched dot product for one Q8_0 block (32 elements).
///
/// input: &[f32] (length 32), quantized: &[i8] (length 32), scale: f32
/// Returns the dot product: sum(input[i] * quantized[i]) * scale
pub fn dot_q8_block(input: &[f32], quantized: &[i8], scale: f32) -> f32 {
    debug_assert!(input.len() >= 32);
    debug_assert!(quantized.len() >= 32);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 feature detected at runtime, input/quantized are valid slices of len >= 32
            return unsafe { x86::dot_q8_block_avx2(input, quantized, scale) };
        }
        // SSE2 is always available on x86_64
        // SAFETY: SSE2 guaranteed on x86_64, input/quantized are valid slices of len >= 32
        return unsafe { x86::dot_q8_block_sse2(input, quantized, scale) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is guaranteed on aarch64, input/quantized are valid slices of len >= 32
        return unsafe { arm::dot_q8_block_neon(input, quantized, scale) };
    }

    #[allow(unreachable_code)]
    dot_q8_block_scalar(input, quantized, scale)
}

/// Runtime-dispatched dot product for one Q4_0 block (32 elements).
///
/// input: &[f32] (length 32), packed: &[u8] (length 16), scale: f32
/// Packed format: low nibble = even index, high nibble = odd index.
/// Returns the dot product: sum(input[i] * dequant(packed[i])) * scale
pub fn dot_q4_block(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    debug_assert!(input.len() >= 32);
    debug_assert!(packed.len() >= 16);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 feature detected at runtime, input/packed are valid slices
            return unsafe { x86::dot_q4_block_avx2(input, packed, scale) };
        }
        // SAFETY: SSE2 guaranteed on x86_64, input/packed are valid slices
        return unsafe { x86::dot_q4_block_sse2(input, packed, scale) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is guaranteed on aarch64, input/packed are valid slices
        return unsafe { arm::dot_q4_block_neon(input, packed, scale) };
    }

    #[allow(unreachable_code)]
    dot_q4_block_scalar(input, packed, scale)
}

/// Scalar-only dot product for Q8_0 block (for testing).
pub fn dot_q8_block_scalar_ref(input: &[f32], quantized: &[i8], scale: f32) -> f32 {
    dot_q8_block_scalar(input, quantized, scale)
}

/// Scalar-only dot product for Q4_0 block (for testing).
pub fn dot_q4_block_scalar_ref(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    dot_q4_block_scalar(input, packed, scale)
}
