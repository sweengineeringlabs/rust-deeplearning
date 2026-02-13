use llmforge::core::tensor::{Tensor, DType};
use std::time::Instant;

fn main() {
    println!("LLM Forge Tensor Benchmark");
    
    // Matrix size: [1024, 1024]
    let m = 1024;
    let k = 1024;
    let n = 1024;
    
    println!("Creating tensors [{}, {}] x [{}, {}]...", m, k, k, n);
    
    // Helper to create random tensor
    let create_random = |rows, cols| {
         let size = rows * cols;
         let mut data = Vec::with_capacity(size * 4);
         for _ in 0..size {
             let val = 1.0f32; // Simplified initialization
             data.extend_from_slice(&val.to_ne_bytes());
         }
         Tensor::new(data, vec![rows, cols], DType::F32)
    };
    
    let a = create_random(m, k);
    let b = create_random(k, n);
    
    println!("Warming up...");
    for _ in 0..5 {
        let _ = a.matmul(&b).unwrap();
    }
    
    println!("Benchmarking matmul...");
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.matmul(&b).unwrap();
    }
    let duration = start.elapsed();
    
    let avg_time = duration.as_secs_f64() / iterations as f64;
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / (avg_time * 1e9);
    
    println!("Avg time: {:.4} seconds", avg_time);
    println!("Performance: {:.2} GFLOPS", gflops);
}
