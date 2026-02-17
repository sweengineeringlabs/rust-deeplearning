/// Benchmark: dequant-to-F32 matmul vs native Q4_0×Q8_0 integer dot product matmul.
///
/// Compares at TinyLlama-relevant dimensions:
///   - Attention projection:  [1, 2048] × [2048, 2048]
///   - FFN up/gate projection: [1, 2048] × [5632, 2048]
///   - Vocab projection:       [1, 2048] × [32000, 2048]  (optional, large)
///
/// Also benchmarks the raw block-level kernel: dot_q4_block vs dot_q4q8_block.
use llmforge::core::tensor::{Tensor, DType};
use llmforge::quantization::{
    quantize_tensor_q4, quantized_matmul_q4, quantized_matmul_q4_native,
    quantize_row_q8_0, simd,
    Q4_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES,
};
use std::time::Instant;

fn make_f32_tensor(rows: usize, cols: usize) -> Tensor {
    // Deterministic pseudo-random data (fast, no rng crate needed at runtime)
    let size = rows * cols;
    let mut data = Vec::with_capacity(size * 4);
    let mut v: u32 = 0xDEAD_BEEF;
    for _ in 0..size {
        v ^= v << 13;
        v ^= v >> 17;
        v ^= v << 5;
        let f = (v as f32) / (u32::MAX as f32) * 2.0 - 1.0; // [-1, 1]
        data.extend_from_slice(&f.to_ne_bytes());
    }
    Tensor::new(data, vec![rows, cols], DType::F32)
}

fn bench_block_kernels() {
    println!("=== Block-level kernel benchmark (single Q4_0 block = 32 elements) ===\n");

    // Set up a single block
    let mut packed = [0u8; 16];
    for i in 0..16 {
        packed[i] = ((i as u8 * 7 + 3) & 0xFF) | (((i as u8 * 11 + 5) & 0x0F) << 4);
    }
    let input_f32: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.05).collect();
    let q8_values: Vec<i8> = (0..32).map(|i| ((i * 7 - 50) % 128) as i8).collect();
    let scale = 0.25f32;

    let iterations = 2_000_000;

    // Warmup
    for _ in 0..1000 {
        std::hint::black_box(simd::dot_q4_block(
            std::hint::black_box(&input_f32),
            std::hint::black_box(&packed),
            std::hint::black_box(scale),
        ));
        std::hint::black_box(simd::dot_q4q8_block(
            std::hint::black_box(&packed),
            std::hint::black_box(&q8_values),
        ));
    }

    // Bench old: dot_q4_block (dequant to F32)
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iterations {
        sink += simd::dot_q4_block(
            std::hint::black_box(&input_f32),
            std::hint::black_box(&packed),
            std::hint::black_box(scale),
        );
    }
    let old_dur = start.elapsed();
    std::hint::black_box(sink);

    // Bench new: dot_q4q8_block (integer)
    let start = Instant::now();
    let mut sink_i = 0i32;
    for _ in 0..iterations {
        sink_i = sink_i.wrapping_add(simd::dot_q4q8_block(
            std::hint::black_box(&packed),
            std::hint::black_box(&q8_values),
        ));
    }
    let new_dur = start.elapsed();
    std::hint::black_box(sink_i);

    let old_ns = old_dur.as_nanos() as f64 / iterations as f64;
    let new_ns = new_dur.as_nanos() as f64 / iterations as f64;
    let speedup = old_ns / new_ns;

    println!("  dot_q4_block   (dequant F32):  {:.1} ns/block", old_ns);
    println!("  dot_q4q8_block (integer i32):  {:.1} ns/block", new_ns);
    println!("  Speedup: {:.2}x", speedup);
    println!();
}

fn bench_matmul(label: &str, batch: usize, in_features: usize, out_features: usize, iterations: usize) {
    println!("--- {} : input [{}, {}] × weights [{}, {}] ---",
        label, batch, in_features, out_features, in_features);

    let input = make_f32_tensor(batch, in_features);
    let weight_f32 = make_f32_tensor(out_features, in_features);
    let weight_q4 = quantize_tensor_q4(&weight_f32).unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = quantized_matmul_q4(&input, &weight_q4).unwrap();
        let _ = quantized_matmul_q4_native(&input, &weight_q4).unwrap();
    }

    // Bench old path (dequant to F32)
    let start = Instant::now();
    for _ in 0..iterations {
        let r = quantized_matmul_q4(std::hint::black_box(&input), std::hint::black_box(&weight_q4)).unwrap();
        std::hint::black_box(&r);
    }
    let old_dur = start.elapsed();

    // Bench new path (native integer)
    let start = Instant::now();
    for _ in 0..iterations {
        let r = quantized_matmul_q4_native(std::hint::black_box(&input), std::hint::black_box(&weight_q4)).unwrap();
        std::hint::black_box(&r);
    }
    let new_dur = start.elapsed();

    let old_ms = old_dur.as_secs_f64() * 1000.0 / iterations as f64;
    let new_ms = new_dur.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = old_ms / new_ms;

    // Numerical comparison
    let old_result = quantized_matmul_q4(&input, &weight_q4).unwrap();
    let new_result = quantized_matmul_q4_native(&input, &weight_q4).unwrap();
    let old_data = old_result.as_slice_f32().unwrap();
    let new_data = new_result.as_slice_f32().unwrap();

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for i in 0..old_data.len() {
        let d = old_data[i];
        let n = new_data[i];
        let abs_err = (d - n).abs();
        if abs_err > max_abs_err { max_abs_err = abs_err; }
        if d.abs() > 1e-8 {
            let rel = abs_err / d.abs();
            if rel > max_rel_err { max_rel_err = rel; }
        }
        sum_sq_err += ((d - n) as f64).powi(2);
        sum_sq_ref += (d as f64).powi(2);
    }
    let rmse = (sum_sq_err / old_data.len() as f64).sqrt();
    let nrmse = if sum_sq_ref > 0.0 { (sum_sq_err / sum_sq_ref).sqrt() } else { 0.0 };

    println!("  Old (dequant F32):   {:.3} ms/iter", old_ms);
    println!("  New (native i32):    {:.3} ms/iter", new_ms);
    println!("  Speedup:             {:.2}x", speedup);
    println!("  Max abs error:       {:.6e}", max_abs_err);
    println!("  Max rel error:       {:.4}%", max_rel_err * 100.0);
    println!("  RMSE:                {:.6e}", rmse);
    println!("  Normalized RMSE:     {:.4}%", nrmse * 100.0);
    println!();
}

fn bench_quantize_row() {
    println!("=== Activation quantization: quantize_row_q8_0 ===\n");

    let in_features = 2048;
    let input_data: Vec<f32> = (0..in_features).map(|i| ((i as f32) - 1024.0) * 0.001).collect();
    let blocks = in_features / Q4_0_BLOCK_SIZE;
    let mut q8_buf = vec![0u8; blocks * Q8_0_BLOCK_BYTES];
    let mut q8_scales = vec![0.0f32; blocks];

    let iterations = 500_000;

    // Warmup
    for _ in 0..1000 {
        quantize_row_q8_0(&input_data, &mut q8_buf, &mut q8_scales);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        quantize_row_q8_0(
            std::hint::black_box(&input_data),
            std::hint::black_box(&mut q8_buf),
            std::hint::black_box(&mut q8_scales),
        );
    }
    let dur = start.elapsed();
    let ns = dur.as_nanos() as f64 / iterations as f64;
    let us = ns / 1000.0;

    println!("  quantize_row_q8_0 ({} elements): {:.1} ns ({:.2} us)", in_features, ns, us);
    println!("  Throughput: {:.1} MB/s", (in_features as f64 * 4.0) / (ns * 1e-9) / 1e6);
    println!();
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Q4_0 MatMul Benchmark: dequant-F32 vs native Q4×Q8 i32   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    bench_block_kernels();
    bench_quantize_row();

    println!("=== Matrix-level benchmarks (TinyLlama dimensions) ===\n");

    // Attention projection: [1, 2048] × [2048, 2048]
    bench_matmul("Attention proj", 1, 2048, 2048, 20);

    // FFN up/gate: [1, 2048] × [5632, 2048]
    bench_matmul("FFN up/gate", 1, 2048, 5632, 10);

    // Smaller test for quick validation
    bench_matmul("Small (validation)", 1, 256, 256, 100);

    // Prefill with sequence length 32: [32, 2048] × [2048, 2048]
    bench_matmul("Prefill (seq=32)", 32, 2048, 2048, 5);

    println!("Done.");
}
