//! Forward pass diagnostic: step through the model layer-by-layer
//! to identify where divergence from llama.cpp begins.
//!
//! GGUF_MODEL_PATH=/tmp/gemma3-gguf/google_gemma-3-1b-it-Q4_0.gguf \
//!   cargo test -p rustml-nlp --release --test forward_diagnostic_test -- --nocapture

use rustml_core::Tensor;
use rustml_gguf::GGUFFile;
use rustml_nlp::{
    convert_tensors, gguf_config_to_model_config, LanguageModel, LlmModel, ModelConfig,
};
use rustml_nn::KVCache;

fn get_model_path() -> Option<String> {
    std::env::var("GGUF_MODEL_PATH").ok()
}

fn load_model(gguf_path: &str) -> Result<(LlmModel, ModelConfig), Box<dyn std::error::Error>> {
    let gguf = GGUFFile::parse_header(gguf_path)?;
    let gguf_config = gguf.to_model_config()?;
    let config = gguf_config_to_model_config(&gguf_config)?;
    let loaded_tensors = gguf.load_and_remap_gemma3(gguf_path, config.n_layers)?;
    let tensors = convert_tensors(loaded_tensors);
    let mut model = LlmModel::from_pretrained_gemma3(&config, tensors)?;
    for layer in &mut model.layers {
        layer.set_native_q4_matmul(true);
    }
    Ok((model, config))
}

fn tensor_stats(t: &Tensor) -> (f32, f32, f32, f32) {
    let data: Vec<f32> = t.iter().collect();
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min, max)
}

fn top_k(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

#[test]
fn diagnose_forward_pass() {
    let gguf_path = match get_model_path() {
        Some(path) => path,
        None => {
            eprintln!("SKIP: Set GGUF_MODEL_PATH");
            return;
        }
    };

    println!("\n=== Forward Pass Diagnostic ===\n");

    let (model, config) = load_model(&gguf_path).expect("Failed to load model");
    let head_dim = config.head_dim.unwrap_or(config.dim / config.n_heads);
    let n_kv_heads = config.n_kv_heads.unwrap_or(config.n_heads);

    // Test 1: BOS only [2]
    println!("--- Test: BOS only [2] ---");
    {
        let input = Tensor::from_vec(vec![2.0f32], vec![1, 1]).unwrap();
        let mut cache = KVCache::new(config.n_layers, config.max_seq_len, head_dim, n_kv_heads);
        let logits = model.forward_with_cache(&input, &mut cache).unwrap();
        let logits_data: Vec<f32> = logits.iter().collect();
        let (mean, std, min, max) = tensor_stats(&logits);
        println!("  Logit stats: mean={:.4} std={:.4} min={:.4} max={:.4}", mean, std, min, max);
        println!("  [REF]        mean=-3.8386 std=2.5344 min=-14.5965 max=11.6090");
        let top = top_k(&logits_data, 5);
        println!("  Top-5: {:?}", top.iter().map(|(i,v)| format!("id={} v={:.3}", i, v)).collect::<Vec<_>>());
        println!("  [REF] Top-5: [1106, 236840, 3617, 140, 3975]");
    }

    // Test 2: BOS + 'The' [2, 818]
    println!("\n--- Test: BOS + 'The' [2, 818] ---");
    {
        let input = Tensor::from_vec(vec![2.0f32, 818.0], vec![1, 2]).unwrap();
        let mut cache = KVCache::new(config.n_layers, config.max_seq_len, head_dim, n_kv_heads);
        let logits = model.forward_with_cache(&input, &mut cache).unwrap();
        // Extract last position logits
        let last = logits.slice(1, 1, 2).unwrap();
        let logits_data: Vec<f32> = last.iter().collect();
        let n = logits_data.len() as f32;
        let mean = logits_data.iter().sum::<f32>() / n;
        let std = (logits_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        let min = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  Logit stats: mean={:.4} std={:.4} min={:.4} max={:.4}", mean, std, min, max);
        println!("  [REF]        mean=-4.4493 std=3.5715 min=-19.9477 max=14.2462");
        let top = top_k(&logits_data, 5);
        println!("  Top-5: {:?}", top.iter().map(|(i,v)| format!("id={} v={:.3}", i, v)).collect::<Vec<_>>());
        println!("  [REF] Top-5: [2608, 861, 236743, 2544, 7384]");
    }

    // Test 3: Process [BOS, 'The'] ONE-AT-A-TIME with cache
    // If this matches reference but all-at-once doesn't, the bug is in multi-token attention
    println!("\n--- Test: [BOS, 'The'] one-at-a-time with cache ---");
    {
        let mut cache = KVCache::new(config.n_layers, config.max_seq_len, head_dim, n_kv_heads);

        // First: process BOS alone
        let bos_input = Tensor::from_vec(vec![2.0f32], vec![1, 1]).unwrap();
        let _ = model.forward_with_cache(&bos_input, &mut cache).unwrap();
        cache.advance(1);

        // Then: process 'The' alone (cache has BOS)
        let the_input = Tensor::from_vec(vec![818.0f32], vec![1, 1]).unwrap();
        let logits = model.forward_with_cache(&the_input, &mut cache).unwrap();
        let logits_data: Vec<f32> = logits.iter().collect();
        let n = logits_data.len() as f32;
        let mean = logits_data.iter().sum::<f32>() / n;
        let std = (logits_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        let min = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  Logit stats: mean={:.4} std={:.4} min={:.4} max={:.4}", mean, std, min, max);
        println!("  [REF]        mean=-4.4493 std=3.5715 min=-19.9477 max=14.2462");
        println!("  [batch2]     mean=-4.7025 std=3.2188 min=-18.4900 max=12.1160  (from above)");
        let top = top_k(&logits_data, 5);
        println!("  Top-5: {:?}", top.iter().map(|(i,v)| format!("id={} v={:.3}", i, v)).collect::<Vec<_>>());
        println!("  [REF] Top-5: [2608, 861, 236743, 2544, 7384]");
    }

    // Test 4: Full 6-token 'The capital of France is' one-at-a-time
    println!("\n--- Test: Full prompt one-at-a-time ---");
    {
        let token_ids = [2u32, 818, 5279, 529, 7001, 563];
        let mut cache = KVCache::new(config.n_layers, config.max_seq_len, head_dim, n_kv_heads);

        for (idx, &tid) in token_ids.iter().enumerate() {
            let input = Tensor::from_vec(vec![tid as f32], vec![1, 1]).unwrap();
            let logits = model.forward_with_cache(&input, &mut cache).unwrap();
            cache.advance(1);

            if idx == token_ids.len() - 1 {
                let logits_data: Vec<f32> = logits.iter().collect();
                let n = logits_data.len() as f32;
                let mean = logits_data.iter().sum::<f32>() / n;
                let std = (logits_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
                let min = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                println!("  After last token (one-at-a-time):");
                println!("    Logit stats: mean={:.4} std={:.4} min={:.4} max={:.4}", mean, std, min, max);
                println!("    [REF]        mean=-5.4143 std=3.4041 min=-19.3257 max=20.0835");
                let top = top_k(&logits_data, 5);
                println!("    Top-5: {:?}", top.iter().map(|(i,v)| format!("id={} v={:.3}", i, v)).collect::<Vec<_>>());
                println!("    [REF] Top-5: [9079(Paris), 5213(**), 496(a), 5633(currently), 5628(located)]");
            }
        }
    }

    // Test 5: Full 6-token all-at-once (for comparison)
    println!("\n--- Test: Full prompt all-at-once ---");
    {
        let token_ids = [2.0f32, 818.0, 5279.0, 529.0, 7001.0, 563.0];
        let input = Tensor::from_vec(token_ids.to_vec(), vec![1, 6]).unwrap();
        let mut cache = KVCache::new(config.n_layers, config.max_seq_len, head_dim, n_kv_heads);
        let logits = model.forward_with_cache(&input, &mut cache).unwrap();
        let last = logits.slice(1, 5, 6).unwrap();
        let logits_data: Vec<f32> = last.iter().collect();
        let n = logits_data.len() as f32;
        let mean = logits_data.iter().sum::<f32>() / n;
        let std = (logits_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        let min = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  After last token (all-at-once):");
        println!("    Logit stats: mean={:.4} std={:.4} min={:.4} max={:.4}", mean, std, min, max);
        println!("    [REF]        mean=-5.4143 std=3.4041 min=-19.3257 max=20.0835");
        let top = top_k(&logits_data, 5);
        println!("    Top-5: {:?}", top.iter().map(|(i,v)| format!("id={} v={:.3}", i, v)).collect::<Vec<_>>());
        println!("    [REF] Top-5: [9079(Paris), 5213(**), 496(a), 5633(currently), 5628(located)]");
    }

    // Test 6: Step through model internals for BOS token
    println!("\n--- Manual layer-by-layer for BOS [2] ---");
    {
        let input = Tensor::from_vec(vec![2.0f32], vec![1, 1]).unwrap();

        // Step 1: Embedding
        let x_emb = model.token_embedding.forward(&input).unwrap();
        let scale = config.embedding_scale.unwrap_or(1.0);
        let x_scaled = x_emb.mul_scalar(scale);
        let (mean, std, min, max) = tensor_stats(&x_scaled);
        println!("  After embedding + scale({:.2}): mean={:.4} std={:.4} min={:.4} max={:.4}",
            scale, mean, std, min, max);

        // Step 2: Layer-by-layer
        let mut cache = KVCache::new(config.n_layers, config.max_seq_len, head_dim, n_kv_heads);
        let mut x = x_scaled;
        for i in 0..config.n_layers {
            x = model.layers[i].forward_with_cache(&x, None, &mut cache, i).unwrap();
            cache.advance(0); // don't advance for prefill, cache handles it

            if i < 3 || i == config.n_layers - 1 || (i + 1) % 6 == 0 {
                let (mean, std, min, max) = tensor_stats(&x);
                let layer_type = if (i + 1) % 6 == 0 { "GLOBAL" } else { "local" };
                println!("  After layer {:>2} [{}]: mean={:.4} std={:.4} min={:.4} max={:.4}",
                    i, layer_type, mean, std, min, max);
            }
        }
        // Advance cache by 1 for the single token
        // Actually for the manual path we need to handle cache properly
        // The cache was not advanced during the loop, let me check...

        // Step 3: Final norm + output projection
        let x_norm = model.norm.forward(&x).unwrap();
        let (mean, std, min, max) = tensor_stats(&x_norm);
        println!("  After final norm: mean={:.4} std={:.4} min={:.4} max={:.4}", mean, std, min, max);

        let logits = model.output.forward(&x_norm).unwrap();
        let logits_data: Vec<f32> = logits.iter().collect();
        let (mean, std, min, max) = tensor_stats(&logits);
        println!("  Final logits: mean={:.4} std={:.4} min={:.4} max={:.4}", mean, std, min, max);
        let top = top_k(&logits_data, 5);
        println!("  Top-5: {:?}", top.iter().map(|(i,v)| format!("id={} v={:.3}", i, v)).collect::<Vec<_>>());
    }
}
