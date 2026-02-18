//! Micro-benchmarks for the three optimizations.
//! Run with: cargo test --release -p rustml-nn --test bench_optimizations -- --nocapture

use std::time::Instant;

#[test]
fn bench_all_optimizations() {
    println!("\n=== Optimization Micro-Benchmarks (--release) ===\n");
    bench_q4_1_kernel();
    bench_rms_norm();
    bench_rope();
    bench_inplace_ops();
    bench_softmax_rayon_threshold();
    bench_batched_matmul_rayon_threshold();
    bench_inplace_score_scaling();
    println!("=== Done ===\n");
}

fn bench_q4_1_kernel() {
    println!("--- 1. Q4_1 x Q8_0 SIMD Kernel ---");

    let mut packed = [0u8; 16];
    for j in 0..16 {
        packed[j] = ((j as u8 + 1) << 4) | (j as u8);
    }
    let q8_values: Vec<i8> = (0..32).map(|i| (i - 16) as i8).collect();

    let iterations = 2_000_000u64;

    // Warmup
    for _ in 0..10_000 {
        std::hint::black_box(rustml_quant::simd::dot_q4_1_q8_block(
            std::hint::black_box(&packed),
            std::hint::black_box(unsafe {
                std::slice::from_raw_parts(q8_values.as_ptr() as *const i8, 32)
            }),
        ));
    }

    // Q4_1 kernel
    let start = Instant::now();
    let mut sink = (0i32, 0i32);
    for _ in 0..iterations {
        let (d, s) = rustml_quant::simd::dot_q4_1_q8_block(
            std::hint::black_box(&packed),
            std::hint::black_box(unsafe {
                std::slice::from_raw_parts(q8_values.as_ptr() as *const i8, 32)
            }),
        );
        sink.0 = sink.0.wrapping_add(d);
        sink.1 = sink.1.wrapping_add(s);
    }
    let dur = start.elapsed();
    std::hint::black_box(sink);
    let ns_q4_1 = dur.as_nanos() as f64 / iterations as f64;

    // Q4_0 kernel for comparison
    let start = Instant::now();
    let mut sink2 = 0i32;
    for _ in 0..iterations {
        sink2 = sink2.wrapping_add(rustml_quant::simd::dot_q4q8_block(
            std::hint::black_box(&packed),
            std::hint::black_box(unsafe {
                std::slice::from_raw_parts(q8_values.as_ptr() as *const i8, 32)
            }),
        ));
    }
    let dur2 = start.elapsed();
    std::hint::black_box(sink2);
    let ns_q4_0 = dur2.as_nanos() as f64 / iterations as f64;

    println!("  dot_q4_1_q8_block: {:.1} ns/block", ns_q4_1);
    println!("  dot_q4q8_block (Q4_0 ref): {:.1} ns/block", ns_q4_0);
    println!("  Q4_1 overhead vs Q4_0: {:.0}% (returns 2 values vs 1)", (ns_q4_1 / ns_q4_0 - 1.0) * 100.0);
    println!();
}

fn bench_rms_norm() {
    println!("--- 2. SIMD RMSNorm (Gemma 3: dim=1152) ---");

    let dim = 1152;
    let x = rustml_core::Tensor::randn(vec![1, 1, dim]);
    let weight = rustml_core::Tensor::ones(vec![dim]);
    let eps = 1e-6f32;

    let iterations = 100_000u64;

    // Warmup
    for _ in 0..1000 {
        let _ = std::hint::black_box(x.rms_norm(&weight, eps).unwrap());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let r = x.rms_norm(&weight, eps).unwrap();
        std::hint::black_box(&r);
    }
    let dur = start.elapsed();
    let us = dur.as_secs_f64() * 1e6 / iterations as f64;
    let per_layer_26 = us * 26.0 * 4.0; // 4 norms per layer (Gemma 3 sandwich)
    println!("  rms_norm [1,1,{}]: {:.2} us/call", dim, us);
    println!("  Per-token overhead (26 layers × 4 norms): {:.0} us = {:.2} ms", per_layer_26, per_layer_26 / 1000.0);
    println!();
}

fn bench_rope() {
    println!("--- 3. SIMD RoPE (Gemma 3: head_dim=256, 4 heads, decode) ---");

    let head_dim = 256;
    let n_heads = 4;
    let seq_len = 1;

    let rope = rustml_nn::RoPEFreqs::new(head_dim, 1024, 10000.0);
    let x = rustml_core::Tensor::randn(vec![1, n_heads, seq_len, head_dim]);

    let iterations = 200_000u64;

    // Warmup
    for _ in 0..1000 {
        let _ = std::hint::black_box(rope.apply(&x, 0).unwrap());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let r = rope.apply(&x, 100).unwrap();
        std::hint::black_box(&r);
    }
    let dur = start.elapsed();
    let us = dur.as_secs_f64() * 1e6 / iterations as f64;
    // 2 RoPE calls per layer (Q and K), 26 layers
    let per_token = us * 26.0 * 2.0;
    println!("  rope.apply [1,{},{},{}]: {:.2} us/call", n_heads, seq_len, head_dim, us);
    println!("  Per-token overhead (26 layers × 2 calls): {:.0} us = {:.2} ms", per_token, per_token / 1000.0);
    println!();
}

fn bench_inplace_ops() {
    println!("--- 4. In-place vs Allocating Add (dim=1152) ---");

    let dim = 1152;
    let iterations = 500_000u64;

    let a = rustml_core::Tensor::randn(vec![1, 1, dim]);
    let b = rustml_core::Tensor::randn(vec![1, 1, dim]);

    // Warmup
    for _ in 0..1000 {
        let _ = std::hint::black_box(a.add(&b).unwrap());
    }

    // Allocating add
    let start = Instant::now();
    for _ in 0..iterations {
        let r = a.add(&b).unwrap();
        std::hint::black_box(&r);
    }
    let alloc_dur = start.elapsed();
    let alloc_ns = alloc_dur.as_nanos() as f64 / iterations as f64;

    // Pure in-place add (reuse same buffer)
    let mut x = rustml_core::Tensor::randn(vec![1, 1, dim]);
    // Warmup
    for _ in 0..1000 {
        x.add_inplace(std::hint::black_box(&b)).unwrap();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        x.add_inplace(std::hint::black_box(&b)).unwrap();
    }
    let inplace_dur = start.elapsed();
    std::hint::black_box(&x);
    let inplace_ns = inplace_dur.as_nanos() as f64 / iterations as f64;

    let speedup = alloc_ns / inplace_ns;
    // Per-token savings: 4 residual adds per layer × 26 layers
    let saved_per_token_us = (alloc_ns - inplace_ns) * 4.0 * 26.0 / 1000.0;
    println!("  Allocating add:  {:.0} ns/call", alloc_ns);
    println!("  In-place add:    {:.0} ns/call", inplace_ns);
    println!("  Speedup: {:.1}x", speedup);
    println!("  Per-token savings (26 layers × 4 adds): {:.0} us = {:.2} ms", saved_per_token_us, saved_per_token_us / 1000.0);
    println!();
}

fn bench_softmax_rayon_threshold() {
    println!("--- 5. Softmax Rayon Threshold (decode attention: [1,H,1,T]) ---");

    let iterations = 100_000u64;

    // Small tensor (sequential path): [1, 32, 1, 64] = 2048 elements (< 4096)
    let small_data: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();
    let small = rustml_core::Tensor::from_vec(small_data, vec![1, 32, 1, 64]).unwrap();

    // Warmup
    for _ in 0..1000 {
        let _ = std::hint::black_box(small.softmax(-1).unwrap());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let r = small.softmax(-1).unwrap();
        std::hint::black_box(&r);
    }
    let small_dur = start.elapsed();
    let small_us = small_dur.as_secs_f64() * 1e6 / iterations as f64;

    // Large tensor (rayon path): [1, 32, 1, 256] = 8192 elements (>= 4096)
    let large_data: Vec<f32> = (0..8192).map(|i| (i as f32 * 0.01).sin()).collect();
    let large = rustml_core::Tensor::from_vec(large_data, vec![1, 32, 1, 256]).unwrap();

    for _ in 0..1000 {
        let _ = std::hint::black_box(large.softmax(-1).unwrap());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let r = large.softmax(-1).unwrap();
        std::hint::black_box(&r);
    }
    let large_dur = start.elapsed();
    let large_us = large_dur.as_secs_f64() * 1e6 / iterations as f64;

    let per_token_small = small_us * 2.0 * 26.0; // 2 softmax per layer (scores + ...), 26 layers
    println!("  Sequential [1,32,1,64]  (2048 elem): {:.2} us/call", small_us);
    println!("  Parallel   [1,32,1,256] (8192 elem): {:.2} us/call", large_us);
    println!("  Per-token decode overhead (26 layers × 2 softmax): {:.0} us = {:.2} ms", per_token_small, per_token_small / 1000.0);
    println!();
}

fn bench_batched_matmul_rayon_threshold() {
    println!("--- 6. Batched Matmul Rayon Threshold (decode: Q@K^T) ---");

    let iterations = 100_000u64;

    // Decode Q@K^T: [32, 1, 64] x [32, 64, 128] -> [32, 1, 128]
    // total output = 32 * 1 * 128 = 4096 (boundary — sequential)
    let q_data: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();
    let kt_data: Vec<f32> = (0..262144).map(|i| (i as f32 * 0.001).sin()).collect();
    let q = rustml_core::Tensor::from_vec(q_data, vec![32, 1, 64]).unwrap();
    let kt = rustml_core::Tensor::from_vec(kt_data, vec![32, 64, 128]).unwrap();

    // Warmup
    for _ in 0..1000 {
        let _ = std::hint::black_box(q.batched_matmul(&kt).unwrap());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let r = q.batched_matmul(&kt).unwrap();
        std::hint::black_box(&r);
    }
    let dur = start.elapsed();
    let us = dur.as_secs_f64() * 1e6 / iterations as f64;

    // Also bench attn @ V: [32, 1, 128] x [32, 128, 64] -> [32, 1, 64]
    // total output = 32 * 1 * 64 = 2048 (sequential)
    let attn_data: Vec<f32> = (0..4096).map(|i| i as f32 * 0.001).collect();
    let v_data: Vec<f32> = (0..262144).map(|i| (i as f32 * 0.001).sin()).collect();
    let attn = rustml_core::Tensor::from_vec(attn_data, vec![32, 1, 128]).unwrap();
    let v = rustml_core::Tensor::from_vec(v_data, vec![32, 128, 64]).unwrap();

    for _ in 0..1000 {
        let _ = std::hint::black_box(attn.batched_matmul(&v).unwrap());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let r = attn.batched_matmul(&v).unwrap();
        std::hint::black_box(&r);
    }
    let dur2 = start.elapsed();
    let us2 = dur2.as_secs_f64() * 1e6 / iterations as f64;

    // Per-token: 2 batched matmuls per layer × 26 layers
    let per_token = (us + us2) * 26.0;
    println!("  Q@K^T [32,1,64]×[32,64,128]:   {:.2} us/call", us);
    println!("  attn@V [32,1,128]×[32,128,64]:  {:.2} us/call", us2);
    println!("  Per-token decode (26 layers × 2 matmuls): {:.0} us = {:.2} ms", per_token, per_token / 1000.0);
    println!();
}

fn bench_inplace_score_scaling() {
    println!("--- 7. In-place Score Scaling (mul_scalar_inplace vs div_scalar) ---");

    let iterations = 500_000u64;
    let scale = (64.0f32).sqrt(); // typical head_dim=64

    // Simulates attention scores: [1, 32, 1, 128]
    let scores_data: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.1) - 200.0).collect();
    let scores_template = rustml_core::Tensor::from_vec(scores_data, vec![1, 32, 1, 128]).unwrap();

    // Warmup
    for _ in 0..1000 {
        let _ = std::hint::black_box(scores_template.div_scalar(scale));
    }

    // Allocating: div_scalar
    let start = Instant::now();
    for _ in 0..iterations {
        let r = scores_template.div_scalar(scale);
        std::hint::black_box(&r);
    }
    let alloc_dur = start.elapsed();
    let alloc_ns = alloc_dur.as_nanos() as f64 / iterations as f64;

    // In-place: mul_scalar_inplace(1/scale) on a fresh clone each time
    // (simulates the real pattern: matmul returns a fresh tensor, then we scale in-place)
    let inv_scale = 1.0 / scale;
    for _ in 0..1000 {
        let mut s = scores_template.clone();
        s.mul_scalar_inplace(inv_scale).unwrap();
        std::hint::black_box(&s);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let mut s = scores_template.clone();
        s.mul_scalar_inplace(inv_scale).unwrap();
        std::hint::black_box(&s);
    }
    let inplace_dur = start.elapsed();
    let inplace_ns = inplace_dur.as_nanos() as f64 / iterations as f64;

    let speedup = alloc_ns / inplace_ns;
    let saved_per_token_us = (alloc_ns - inplace_ns) * 26.0 / 1000.0; // 1 per layer × 26 layers
    println!("  div_scalar (alloc):        {:.0} ns/call", alloc_ns);
    println!("  mul_scalar_inplace:        {:.0} ns/call", inplace_ns);
    println!("  Speedup: {:.1}x", speedup);
    println!("  Per-token savings (26 layers × 1 scale): {:.0} us = {:.2} ms", saved_per_token_us, saved_per_token_us / 1000.0);
    println!();
}
