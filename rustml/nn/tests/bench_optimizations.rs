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
