//! Logit diagnostic tool for comparing LLMForge output against llama.cpp.
//!
//! Loads TinyLlama-1.1B Q4_0, runs a short prompt with greedy decoding,
//! and dumps token IDs + top-20 logits as JSON for prefill and 3 decode steps.
//!
//! Usage:
//! ```bash
//! # Auto-download TinyLlama Q4_0 and tokenizer
//! cargo run --release --example logit_dump
//!
//! # Use a local GGUF file
//! cargo run --release --example logit_dump -- /path/to/model.gguf [/path/to/tokenizer.json]
//!
//! # Custom prompt
//! cargo run --release --example logit_dump -- --prompt "The capital of France"
//! ```
//!
//! Output: JSON to stdout (diagnostics to stderr) for easy piping:
//! ```bash
//! cargo run --release --example logit_dump 2>/dev/null > logits_rust.json
//! ```

use llmforge::core::tensor::{Tensor, DType};
use llmforge::loader::ModelLoader;
use llmforge::models::LlmModel;
use llmforge::tokenization::{HFTokenizer, Tokenizer};
use llmforge::attention::KVCache;

use std::path::PathBuf;

// ── Auto-download defaults ───────────────────────────────────────────
const DEFAULT_GGUF_REPO: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
const DEFAULT_GGUF_FILE: &str = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
const DEFAULT_TOKENIZER_REPO: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
const DEFAULT_PROMPT: &str = "The capital of France is";
const DECODE_STEPS: usize = 3;
const TOP_N: usize = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut gguf_path_arg: Option<String> = None;
    let mut tokenizer_path_arg: Option<String> = None;
    let mut prompt = DEFAULT_PROMPT.to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                eprintln!("Usage: {} [gguf_path] [tokenizer_path] [--prompt TEXT]", args[0]);
                eprintln!();
                eprintln!("Dumps top-{} logits as JSON for prefill + {} decode steps.", TOP_N, DECODE_STEPS);
                eprintln!("Diagnostics go to stderr, JSON to stdout.");
                std::process::exit(0);
            }
            "--prompt" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    prompt = p.clone();
                }
            }
            other if !other.starts_with("--") => {
                if gguf_path_arg.is_none() {
                    gguf_path_arg = Some(other.to_string());
                } else if tokenizer_path_arg.is_none() {
                    tokenizer_path_arg = Some(other.to_string());
                }
            }
            _ => {
                eprintln!("Warning: unknown flag '{}'", args[i]);
            }
        }
        i += 1;
    }

    // ── Resolve paths ────────────────────────────────────────────────
    let (gguf_path, tokenizer_path): (PathBuf, PathBuf) = if let Some(path) = gguf_path_arg {
        let tok = tokenizer_path_arg.map(PathBuf::from)
            .expect("When providing a GGUF path, also provide a tokenizer path");
        (PathBuf::from(&path), tok)
    } else {
        eprintln!("Auto-downloading TinyLlama-1.1B-Chat Q4_0...");
        let api = hf_hub::api::sync::Api::new()?;

        let gguf = api.model(DEFAULT_GGUF_REPO.to_string()).get(DEFAULT_GGUF_FILE)?;
        eprintln!("  GGUF: {}", gguf.display());

        let tok = api.model(DEFAULT_TOKENIZER_REPO.to_string()).get("tokenizer.json")?;
        eprintln!("  Tokenizer: {}", tok.display());

        (gguf, tok)
    };

    // ── Load model ───────────────────────────────────────────────────
    eprintln!("Loading GGUF: {}", gguf_path.display());
    let (config, weights) = ModelLoader::load_gguf(&gguf_path)?;

    eprintln!("Building model (dim={}, layers={}, heads={}, kv_heads={})...",
        config.dim, config.n_layers, config.n_heads,
        config.n_kv_heads.unwrap_or(config.n_heads));
    let model = LlmModel::from_pretrained(&config, weights)?;

    // ── Load tokenizer ───────────────────────────────────────────────
    eprintln!("Loading tokenizer: {}", tokenizer_path.display());
    let tokenizer = HFTokenizer::from_file(&tokenizer_path)?;

    // ── Encode prompt (raw, no chat template) ────────────────────────
    let mut token_ids = tokenizer.encode(&prompt)?;

    // Prepend BOS if configured
    if let Some(bos) = config.bos_token_id {
        token_ids.insert(0, bos);
    }

    eprintln!("Prompt: {:?}", prompt);
    eprintln!("Token IDs ({}): {:?}", token_ids.len(), token_ids);

    // Decode each token for verification
    for &tid in &token_ids {
        if let Ok(text) = tokenizer.decode(&[tid]) {
            eprintln!("  token {} -> {:?}", tid, text);
        }
    }

    // ── Create KV cache ──────────────────────────────────────────────
    let head_dim = model.d_model / model.n_heads;
    let mut cache = KVCache::new(
        model.n_layers,
        model.max_seq_len,
        head_dim,
        model.n_kv_heads,
    );

    // ── Prefill ──────────────────────────────────────────────────────
    eprintln!("\n--- Prefill ({} tokens) ---", token_ids.len());

    let seq_len = token_ids.len();
    let mut input_data = Vec::with_capacity(seq_len);
    for &t in &token_ids {
        input_data.push(t as f32);
    }
    let input_bytes = llmforge::core::tensor::f32_vec_to_bytes(input_data);
    let input = Tensor::new(input_bytes, vec![1, seq_len], DType::F32);

    let logits_tensor = model.forward_with_cache(&input, &mut cache)?;
    cache.advance(seq_len);

    let logits_data = logits_tensor.as_slice_f32()?;
    let vocab_size = model.vocab_size;
    let start = (seq_len - 1) * vocab_size;
    let prefill_logits = &logits_data[start..start + vocab_size];

    let prefill_top = top_n_logits(prefill_logits, TOP_N);
    let prefill_token = prefill_top[0].0;

    eprintln!("Prefill top-1: token {} = {:?} (logit {:.4})",
        prefill_token,
        tokenizer.decode(&[prefill_token]).unwrap_or_default(),
        prefill_top[0].1);

    // ── Decode steps ─────────────────────────────────────────────────
    let mut decode_results = Vec::new();
    let mut current_token = prefill_token;

    for step in 0..DECODE_STEPS {
        let input_val = current_token as f32;
        let input_bytes = input_val.to_ne_bytes().to_vec();
        let input = Tensor::new(input_bytes, vec![1, 1], DType::F32);

        let logits_tensor = model.forward_with_cache(&input, &mut cache)?;
        cache.advance(1);

        let logits_data = logits_tensor.as_slice_f32()?;
        let step_logits = &logits_data[0..vocab_size];

        let step_top = top_n_logits(step_logits, TOP_N);
        let next_token = step_top[0].0;

        // Logit statistics for debugging
        let logit_mean = step_logits.iter().sum::<f32>() / step_logits.len() as f32;
        let logit_std = (step_logits.iter().map(|x| (x - logit_mean).powi(2)).sum::<f32>()
            / step_logits.len() as f32).sqrt();
        let logit_min = step_logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let logit_max = step_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        eprintln!("Decode step {}: input token {} -> top-1 token {} = {:?} (logit {:.4})",
            step, current_token, next_token,
            tokenizer.decode(&[next_token]).unwrap_or_default(),
            step_top[0].1);
        eprintln!("  logit stats: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            logit_mean, logit_std, logit_min, logit_max);

        decode_results.push((current_token, step_top));
        current_token = next_token;
    }

    // ── Build JSON output ────────────────────────────────────────────
    let mut json = String::from("{\n");

    // Prompt info
    json.push_str(&format!("  \"prompt\": {:?},\n", prompt));
    json.push_str(&format!("  \"token_ids\": {:?},\n", token_ids));
    json.push_str(&format!("  \"vocab_size\": {},\n", vocab_size));

    // Model info
    json.push_str(&format!("  \"model\": {{\n"));
    json.push_str(&format!("    \"dim\": {},\n", config.dim));
    json.push_str(&format!("    \"n_layers\": {},\n", config.n_layers));
    json.push_str(&format!("    \"n_heads\": {},\n", config.n_heads));
    json.push_str(&format!("    \"n_kv_heads\": {},\n", config.n_kv_heads.unwrap_or(config.n_heads)));
    json.push_str(&format!("    \"rope_theta\": {}\n", config.rope_theta));
    json.push_str("  },\n");

    // Prefill logits
    json.push_str("  \"prefill\": {\n");
    json.push_str(&format!("    \"input_tokens\": {:?},\n", token_ids));
    json.push_str("    \"top_logits\": [\n");
    for (i, (tid, val)) in prefill_top.iter().enumerate() {
        let token_str = tokenizer.decode(&[*tid]).unwrap_or_default();
        let comma = if i + 1 < prefill_top.len() { "," } else { "" };
        json.push_str(&format!("      {{\"rank\": {}, \"token_id\": {}, \"token\": {:?}, \"logit\": {:.6}}}{}\n",
            i, tid, token_str, val, comma));
    }
    json.push_str("    ]\n");
    json.push_str("  },\n");

    // Decode steps
    json.push_str("  \"decode_steps\": [\n");
    for (step_idx, (input_token, top)) in decode_results.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"step\": {},\n", step_idx));
        json.push_str(&format!("      \"input_token\": {},\n", input_token));
        json.push_str("      \"top_logits\": [\n");
        for (i, (tid, val)) in top.iter().enumerate() {
            let token_str = tokenizer.decode(&[*tid]).unwrap_or_default();
            let comma = if i + 1 < top.len() { "," } else { "" };
            json.push_str(&format!("        {{\"rank\": {}, \"token_id\": {}, \"token\": {:?}, \"logit\": {:.6}}}{}\n",
                i, tid, token_str, val, comma));
        }
        json.push_str("      ]\n");
        let step_comma = if step_idx + 1 < decode_results.len() { "," } else { "" };
        json.push_str(&format!("    }}{}\n", step_comma));
    }
    json.push_str("  ]\n");

    json.push_str("}\n");

    // Print JSON to stdout
    print!("{}", json);

    eprintln!("\nDone. Pipe stdout to a file: cargo run --release --example logit_dump 2>/dev/null > logits_rust.json");

    Ok(())
}

/// Extract top-N (token_id, logit_value) pairs sorted by logit descending.
fn top_n_logits(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> = logits.iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    // Sort descending by logit value
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(n);
    indexed
}
