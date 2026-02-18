//! Gemma 3 GGUF inference example.
//!
//! Loads a quantized Gemma 3 1B-IT model from GGUF format and runs text generation.
//!
//! Usage:
//!   cargo run -p rustml-nlp --release --example gemma3_gguf -- /path/to/model.gguf

use rustml_core::Tensor;
use rustml_gguf::GGUFFile;
use rustml_nlp::{
    convert_tensors, gguf_config_to_model_config, LlmModel, LanguageModel,
};
use rustml_tokenizer::{GgufTokenizer, Tokenizer};
use rustml_nn::KVCache;
use std::time::Instant;

/// Greedy decode: generate tokens until EOS/EOT or max tokens reached.
fn generate(
    model: &LlmModel,
    cache: &mut KVCache,
    tokenizer: &GgufTokenizer,
    token_ids: &[u32],
    max_new_tokens: usize,
    stop_tokens: &[u32],
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut generated = token_ids.to_vec();

    // Prefill
    let t0 = Instant::now();
    let input_data: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
    let input = Tensor::from_vec(input_data, vec![1, token_ids.len()])?;
    let logits = model.forward_with_cache(&input, cache)?;
    cache.advance(token_ids.len());

    let last_logits = logits.slice(1, token_ids.len() - 1, token_ids.len())?;
    let logits_data: Vec<f32> = last_logits.iter().collect();
    let next_token = argmax(&logits_data);
    generated.push(next_token);

    let prefill_time = t0.elapsed().as_secs_f32();
    println!(
        "  Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        token_ids.len(), prefill_time, token_ids.len() as f32 / prefill_time
    );

    // Print top-5 first predictions
    let mut indexed: Vec<(usize, f32)> = logits_data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("  Top-5 predictions:");
    for &(idx, logit) in indexed.iter().take(5) {
        let tok = tokenizer.decode(&[idx as u32]).unwrap_or_default();
        println!("    id={:<6} logit={:>7.2}  {:?}", idx, logit, tok.trim());
    }

    // Autoregressive decode
    let decode_start = Instant::now();
    for step in 0..max_new_tokens - 1 {
        let last = *generated.last().unwrap();
        if stop_tokens.contains(&last) {
            println!("  [Stop token at step {}]", step);
            break;
        }

        let input = Tensor::from_vec(vec![last as f32], vec![1, 1])?;
        let logits = model.forward_with_cache(&input, cache)?;
        cache.advance(1);

        let logits_data: Vec<f32> = logits.iter().collect();
        generated.push(argmax(&logits_data));
    }

    let new_tokens = generated.len() - token_ids.len();
    let decode_time = decode_start.elapsed().as_secs_f32();
    let tps = if decode_time > 0.01 { new_tokens as f32 / decode_time } else { 0.0 };
    println!("  Decode: {} tokens in {:.2}s ({:.1} tok/s)", new_tokens, decode_time, tps);

    Ok(generated)
}

fn argmax(data: &[f32]) -> u32 {
    data.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/gemma3-gguf/gemma-3-1b-it-Q4_0.gguf".to_string());

    println!("=== Gemma 3 GGUF Inference ===\n");

    // 1. Parse GGUF header
    println!("Loading model: {}", gguf_path);
    let t0 = Instant::now();
    let gguf = GGUFFile::parse_header(&gguf_path)?;
    println!("  GGUF v{}, {} tensors ({:.1}s)", gguf.version, gguf.tensor_infos.len(), t0.elapsed().as_secs_f32());

    // 2. Extract config
    let gguf_config = gguf.to_model_config()?;
    let config = gguf_config_to_model_config(&gguf_config)?;
    println!(
        "  arch={}, dim={}, layers={}, heads={}/{:?}, head_dim={:?}, vocab={}",
        gguf_config.architecture, config.dim, config.n_layers,
        config.n_heads, config.n_kv_heads, config.head_dim, config.vocab_size
    );

    // 3. Build tokenizer
    let tokenizer = GgufTokenizer::from_gguf(&gguf)?;
    println!("  Tokenizer: {} tokens", tokenizer.vocab_size());

    // 4. Load tensors
    let t0 = Instant::now();
    let loaded_tensors = gguf.load_and_remap_gemma3(&gguf_path, config.n_layers)?;
    println!("  Tensors loaded ({:.1}s)", t0.elapsed().as_secs_f32());
    let tensors = convert_tensors(loaded_tensors);

    // 5. Build model
    let t0 = Instant::now();
    let mut model = LlmModel::from_pretrained_gemma3(&config, tensors)?;
    let (total_params, _) = model.parameter_count();
    println!("  Model built: {:.1}M params ({:.1}s)", total_params as f64 / 1e6, t0.elapsed().as_secs_f32());

    for layer in &mut model.layers {
        layer.set_native_q4_matmul(true);
    }

    let head_dim = config.head_dim.unwrap_or(config.dim / config.n_heads);
    let n_kv_heads = config.n_kv_heads.unwrap_or(config.n_heads);
    let bos_id = config.bos_token_id.unwrap_or(2);
    let eos_id = config.eos_token_id.unwrap_or(1);
    let end_of_turn = tokenizer.token_to_id("<end_of_turn>").unwrap_or(107);

    // ============================================================
    // Test 1: Plain text completion
    // ============================================================
    println!("\n--- Test 1: Text Completion ---");
    let prompt = "The capital of France is";
    println!("Prompt: {:?}", prompt);

    let mut token_ids = vec![bos_id];
    token_ids.extend(tokenizer.encode(prompt)?);
    println!("  Token IDs: {:?}", &token_ids);

    let mut cache = KVCache::new(config.n_layers, config.max_seq_len, head_dim, n_kv_heads);
    let generated = generate(
        &model, &mut cache, &tokenizer, &token_ids,
        32, &[eos_id, end_of_turn],
    )?;

    let output = tokenizer.decode(&generated)?;
    println!("\n  Output: {}", output);

    // ============================================================
    // Test 2: Chat template (IT model)
    // ============================================================
    println!("\n--- Test 2: Chat (IT Format) ---");
    let user_msg = "What is the capital of France? Answer in one sentence.";
    println!("User: {:?}", user_msg);

    // Build chat tokens manually with special token IDs
    // Format: <bos><start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n
    let start_of_turn = tokenizer.token_to_id("<start_of_turn>").unwrap_or(106);
    let newline_id = tokenizer.token_to_id("\n");

    let mut chat_ids = vec![bos_id, start_of_turn];
    // "user" without SentencePiece space: look up raw token
    if let Some(user_id) = tokenizer.token_to_id("user") {
        chat_ids.push(user_id);
    } else {
        chat_ids.extend(tokenizer.encode("user")?);
    }
    if let Some(nl) = newline_id { chat_ids.push(nl); }
    chat_ids.extend(tokenizer.encode(user_msg)?);
    chat_ids.push(end_of_turn);
    if let Some(nl) = newline_id { chat_ids.push(nl); }
    chat_ids.push(start_of_turn);
    if let Some(model_id) = tokenizer.token_to_id("model") {
        chat_ids.push(model_id);
    } else {
        chat_ids.extend(tokenizer.encode("model")?);
    }
    if let Some(nl) = newline_id { chat_ids.push(nl); }

    println!("  Chat IDs: {:?} (len={})", &chat_ids, chat_ids.len());

    let mut cache2 = KVCache::new(config.n_layers, config.max_seq_len, head_dim, n_kv_heads);
    let generated2 = generate(
        &model, &mut cache2, &tokenizer, &chat_ids,
        64, &[eos_id, end_of_turn],
    )?;

    // Decode just the generated portion
    let response_ids = &generated2[chat_ids.len()..];
    let response = tokenizer.decode(response_ids)?;
    println!("\n  Model: {}", response.trim());

    println!("\n=== Done ===");
    Ok(())
}
