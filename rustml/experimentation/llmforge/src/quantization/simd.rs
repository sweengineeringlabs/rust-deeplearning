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
/// Packed format: low nibble → positions 0..15, high nibble → positions 16..31.
/// Values stored as unsigned 0-15, subtract 8 to dequant.
fn dot_q4_block_scalar(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..16 {
        let byte = packed[j];
        let lo = (byte & 0x0F) as i8 - 8;
        let hi = ((byte >> 4) & 0x0F) as i8 - 8;
        sum += input[j] * (lo as f32);
        sum += input[j + 16] * (hi as f32);
    }
    sum * scale
}

/// Scalar integer dot product: Q4_0 packed nibbles × Q8_0 i8 values → i32.
/// Returns raw i32 sum; caller applies `(result as f32) * q8_scale * q4_scale`.
/// Packed format: byte j low nibble → position j (0..15), high nibble → position j+16 (16..31).
fn dot_q4q8_block_scalar(packed_q4: &[u8], q8_values: &[i8]) -> i32 {
    let mut sum = 0i32;
    for j in 0..16 {
        let lo = (packed_q4[j] & 0x0F) as i8 - 8;
        let hi = ((packed_q4[j] >> 4) & 0x0F) as i8 - 8;
        sum += (lo as i32) * (q8_values[j] as i32);
        sum += (hi as i32) * (q8_values[j + 16] as i32);
    }
    sum
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

        // Unpack all 32 values: lo nibbles → 0..15, hi nibbles → 16..31
        let mut unpacked = [0.0f32; 32];
        for j in 0..16 {
            let byte = packed[j];
            unpacked[j] = ((byte & 0x0F) as i8 - 8) as f32;
            unpacked[j + 16] = (((byte >> 4) & 0x0F) as i8 - 8) as f32;
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

    /// AVX2 integer dot product: Q4_0 × Q8_0 → i32.
    /// Uses _mm256_maddubs_epi16 trick: compute biased dot (unsigned nibbles × signed q8),
    /// then subtract 8 * sum(q8) to correct for the unsigned→signed offset.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_q4q8_block_avx2(packed_q4: &[u8], q8_values: &[i8]) -> i32 {
        use std::arch::x86_64::*;

        // Load 16 packed bytes
        let packed = _mm_loadu_si128(packed_q4.as_ptr() as *const __m128i);
        let mask_0f = _mm_set1_epi8(0x0F);

        // Extract lo nibbles (positions 0-15) and hi nibbles (positions 16-31)
        let lo = _mm_and_si128(packed, mask_0f);
        let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_0f);

        // Combine into 32 × u8: lo in lanes 0-15, hi in lanes 16-31
        let q4_unsigned = _mm256_set_m128i(hi, lo);

        // Load 32 × i8 Q8 values
        let q8 = _mm256_loadu_si256(q8_values.as_ptr() as *const __m256i);

        // maddubs: unsigned × signed → i16 with adjacent pair addition
        // Computes (q4_raw * q8) per pair, which equals (q4_signed + 8) * q8
        let biased_products = _mm256_maddubs_epi16(q4_unsigned, q8);

        // Pairwise sum i16 → i32: madd with ones
        let ones_i16 = _mm256_set1_epi16(1);
        let biased_i32 = _mm256_madd_epi16(biased_products, ones_i16);

        // Horizontal reduce 8 × i32 → scalar
        let hi128 = _mm256_extracti128_si256(biased_i32, 1);
        let lo128 = _mm256_castsi256_si128(biased_i32);
        let sum128 = _mm_add_epi32(lo128, hi128);
        let hi64 = _mm_srli_si128(sum128, 8);
        let sum64 = _mm_add_epi32(sum128, hi64);
        let hi32 = _mm_srli_si128(sum64, 4);
        let sum32 = _mm_add_epi32(sum64, hi32);
        let biased_dot = _mm_cvtsi128_si32(sum32);

        // Compute 8 × sum(q8) correction:
        // maddubs(ones_u8, q8) = 1*q8[2k] + 1*q8[2k+1] per pair → 16 × i16
        let ones_u8 = _mm256_set1_epi8(1);
        let q8_pairwise = _mm256_maddubs_epi16(ones_u8, q8);
        let q8_i32 = _mm256_madd_epi16(q8_pairwise, ones_i16);

        // Horizontal reduce for q8 sum
        let q8_hi128 = _mm256_extracti128_si256(q8_i32, 1);
        let q8_lo128 = _mm256_castsi256_si128(q8_i32);
        let q8_sum128 = _mm_add_epi32(q8_lo128, q8_hi128);
        let q8_hi64 = _mm_srli_si128(q8_sum128, 8);
        let q8_sum64 = _mm_add_epi32(q8_sum128, q8_hi64);
        let q8_hi32 = _mm_srli_si128(q8_sum64, 4);
        let q8_sum32 = _mm_add_epi32(q8_sum64, q8_hi32);
        let q8_sum = _mm_cvtsi128_si32(q8_sum32);

        biased_dot - 8 * q8_sum
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

        // Unpack all 32 values: lo nibbles → 0..15, hi nibbles → 16..31
        let mut unpacked = [0.0f32; 32];
        for j in 0..16 {
            let byte = packed[j];
            unpacked[j] = ((byte & 0x0F) as i8 - 8) as f32;
            unpacked[j + 16] = (((byte >> 4) & 0x0F) as i8 - 8) as f32;
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
    /// SSE2 integer dot product: Q4_0 × Q8_0 → i32.
    /// No maddubs (SSSE3), so unpacks nibbles to i8 on stack, sign-extends to i16,
    /// and uses _mm_madd_epi16 in 4 chunks of 8 elements.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn dot_q4q8_block_sse2(packed_q4: &[u8], q8_values: &[i8]) -> i32 {
        use std::arch::x86_64::*;

        // Unpack all 32 Q4 values to i8
        let mut unpacked = [0i8; 32];
        for j in 0..16 {
            let byte = packed_q4[j];
            unpacked[j] = (byte & 0x0F) as i8 - 8;
            unpacked[j + 16] = ((byte >> 4) & 0x0F) as i8 - 8;
        }

        let zero = _mm_setzero_si128();
        let mut acc = _mm_setzero_si128(); // 4 × i32 accumulator

        // Process 32 elements in 4 chunks of 8
        for chunk in 0..4 {
            let base = chunk * 8;

            // Load 8 × i8 Q4 values into low 64 bits
            let q4_raw = _mm_loadl_epi64(unpacked.as_ptr().add(base) as *const __m128i);
            // Sign-extend i8 → i16: interleave with sign mask
            let q4_sign = _mm_cmpgt_epi8(zero, q4_raw);
            let q4_i16 = _mm_unpacklo_epi8(q4_raw, q4_sign);

            // Load 8 × i8 Q8 values
            let q8_raw = _mm_loadl_epi64(q8_values.as_ptr().add(base) as *const __m128i);
            let q8_sign = _mm_cmpgt_epi8(zero, q8_raw);
            let q8_i16 = _mm_unpacklo_epi8(q8_raw, q8_sign);

            // madd: i16×i16→i32 with adjacent pair accumulation
            let prod = _mm_madd_epi16(q4_i16, q8_i16);
            acc = _mm_add_epi32(acc, prod);
        }

        // Horizontal reduce 4 × i32 → scalar
        let hi64 = _mm_srli_si128(acc, 8);
        let sum64 = _mm_add_epi32(acc, hi64);
        let hi32 = _mm_srli_si128(sum64, 4);
        let sum32 = _mm_add_epi32(sum64, hi32);
        _mm_cvtsi128_si32(sum32)
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
        for j in 0..16 {
            let byte = packed[j];
            unpacked[j] = ((byte & 0x0F) as i8 - 8) as f32;
            unpacked[j + 16] = (((byte >> 4) & 0x0F) as i8 - 8) as f32;
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
    /// NEON integer dot product: Q4_0 × Q8_0 → i32.
    /// Extracts nibbles, converts to i8, uses vmull_s8 (widening multiply)
    /// and vpadalq_s16 (pairwise add-accumulate i16→i32).
    pub(super) unsafe fn dot_q4q8_block_neon(packed_q4: &[u8], q8_values: &[i8]) -> i32 {
        use std::arch::aarch64::*;

        // Load 16 packed bytes
        let packed = vld1q_u8(packed_q4.as_ptr());
        let mask_0f = vdupq_n_u8(0x0F);
        let offset_8 = vdupq_n_u8(8);

        // Extract nibbles as unsigned
        let lo_u8 = vandq_u8(packed, mask_0f);
        let hi_u8 = vshrq_n_u8(packed, 4);

        // Convert to signed: subtract 8 (wrapping), reinterpret as i8
        let lo_i8 = vreinterpretq_s8_u8(vsubq_u8(lo_u8, offset_8));
        let hi_i8 = vreinterpretq_s8_u8(vsubq_u8(hi_u8, offset_8));

        // Split into 8-element halves
        let lo_low = vget_low_s8(lo_i8);   // positions 0-7
        let lo_high = vget_high_s8(lo_i8);  // positions 8-15
        let hi_low = vget_low_s8(hi_i8);   // positions 16-23
        let hi_high = vget_high_s8(hi_i8);  // positions 24-31

        // Load Q8 values
        let q8_0_15 = vld1q_s8(q8_values.as_ptr());
        let q8_16_31 = vld1q_s8(q8_values.as_ptr().add(16));
        let q8_0_7 = vget_low_s8(q8_0_15);
        let q8_8_15 = vget_high_s8(q8_0_15);
        let q8_16_23 = vget_low_s8(q8_16_31);
        let q8_24_31 = vget_high_s8(q8_16_31);

        // Widening multiply and pairwise accumulate: i8×i8→i16, pairs→i32
        let mut acc = vdupq_n_s32(0);
        acc = vpadalq_s16(acc, vmull_s8(lo_low, q8_0_7));      // positions 0-7
        acc = vpadalq_s16(acc, vmull_s8(lo_high, q8_8_15));     // positions 8-15
        acc = vpadalq_s16(acc, vmull_s8(hi_low, q8_16_23));     // positions 16-23
        acc = vpadalq_s16(acc, vmull_s8(hi_high, q8_24_31));    // positions 24-31

        // Reduce 4 × i32 → scalar
        vaddvq_s32(acc)
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
/// Packed format: byte j low nibble → position j, high nibble → position j+16.
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

/// Runtime-dispatched integer dot product: Q4_0 packed nibbles × Q8_0 i8 values → i32.
///
/// packed_q4: &[u8] (length 16), q8_values: &[i8] (length 32)
/// Returns raw i32 dot product. Caller applies: `(result as f32) * q8_scale * q4_scale`.
/// Integer accumulation within the block eliminates per-element F32 rounding,
/// matching llama.cpp's native quantized matmul behavior.
pub fn dot_q4q8_block(packed_q4: &[u8], q8_values: &[i8]) -> i32 {
    debug_assert!(packed_q4.len() >= 16);
    debug_assert!(q8_values.len() >= 32);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 feature detected at runtime, packed_q4/q8_values are valid slices
            return unsafe { x86::dot_q4q8_block_avx2(packed_q4, q8_values) };
        }
        // SAFETY: SSE2 guaranteed on x86_64, packed_q4/q8_values are valid slices
        return unsafe { x86::dot_q4q8_block_sse2(packed_q4, q8_values) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is guaranteed on aarch64, packed_q4/q8_values are valid slices
        return unsafe { arm::dot_q4q8_block_neon(packed_q4, q8_values) };
    }

    #[allow(unreachable_code)]
    dot_q4q8_block_scalar(packed_q4, q8_values)
}

/// Scalar-only dot product for Q8_0 block (for testing).
pub fn dot_q8_block_scalar_ref(input: &[f32], quantized: &[i8], scale: f32) -> f32 {
    dot_q8_block_scalar(input, quantized, scale)
}

/// Scalar-only dot product for Q4_0 block (for testing).
pub fn dot_q4_block_scalar_ref(input: &[f32], packed: &[u8], scale: f32) -> f32 {
    dot_q4_block_scalar(input, packed, scale)
}

/// Scalar-only integer dot product for Q4_0 × Q8_0 block (for testing).
pub fn dot_q4q8_block_scalar_ref(packed_q4: &[u8], q8_values: &[i8]) -> i32 {
    dot_q4q8_block_scalar(packed_q4, q8_values)
}
