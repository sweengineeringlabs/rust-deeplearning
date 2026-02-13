//! GGUF model inference example.
//!
//! Loads a GGUF file (Llama-style) and runs interactive generation.
//! Can auto-download TinyLlama-1.1B Q4_0 from HuggingFace, or load a local file.
//!
//! ```bash
//! # Auto-download TinyLlama-1.1B-Chat Q4_0 (~670MB)
//! cargo run --example gguf_inference
//!
//! # Load a local GGUF file
//! cargo run --example gguf_inference -- /path/to/model.gguf [/path/to/tokenizer.json]
//! ```

use llmforge::core::tensor::DType;
use llmforge::inference::Generator;
use llmforge::loader::ModelLoader;
use llmforge::models::LlmModel;
use llmforge::tokenization::{HFTokenizer, NaiveTokenizer, Tokenizer};

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;

// ── Generation defaults ──────────────────────────────────────────────
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_K: usize = 40;
const DEFAULT_TOP_P: f32 = 0.9;
const DEFAULT_MAX_TOKENS: usize = 128;

// ── Auto-download defaults ───────────────────────────────────────────
const DEFAULT_GGUF_REPO: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
const DEFAULT_GGUF_FILE: &str = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
const DEFAULT_TOKENIZER_REPO: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // ── Parse CLI arguments ──────────────────────────────────────────
    let mut gguf_path_arg: Option<String> = None;
    let mut tokenizer_path_arg: Option<String> = None;
    let mut temperature = DEFAULT_TEMPERATURE;
    let mut max_tokens = DEFAULT_MAX_TOKENS;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                eprintln!("Usage: {} [gguf_path] [tokenizer_path] [--temp N] [--max-tokens N]", args[0]);
                eprintln!();
                eprintln!("With no arguments, auto-downloads TinyLlama-1.1B-Chat Q4_0 from HuggingFace.");
                eprintln!();
                eprintln!("Examples:");
                eprintln!("  {}                                    # auto-download TinyLlama", args[0]);
                eprintln!("  {} model.gguf                         # local GGUF", args[0]);
                eprintln!("  {} model.gguf tokenizer.json          # local GGUF + tokenizer", args[0]);
                eprintln!("  {} model.gguf --temp 0.9 --max-tokens 256", args[0]);
                std::process::exit(0);
            }
            "--temp" => {
                i += 1;
                temperature = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(temperature);
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(max_tokens);
            }
            other if !other.starts_with("--") => {
                if gguf_path_arg.is_none() {
                    gguf_path_arg = Some(other.to_string());
                } else if tokenizer_path_arg.is_none() {
                    tokenizer_path_arg = Some(other.to_string());
                } else {
                    eprintln!("Warning: unexpected argument '{}'", other);
                }
            }
            _ => {
                eprintln!("Warning: unknown flag '{}'", args[i]);
            }
        }
        i += 1;
    }

    println!("=== LLMForge: GGUF Inference ===\n");

    // ── Resolve model and tokenizer paths ────────────────────────────
    let (gguf_path, tokenizer_path): (PathBuf, Option<PathBuf>) = if let Some(path) = gguf_path_arg {
        (PathBuf::from(&path), tokenizer_path_arg.map(PathBuf::from))
    } else {
        // Auto-download from HuggingFace
        println!("No GGUF path provided — auto-downloading TinyLlama-1.1B-Chat Q4_0...");
        let api = hf_hub::api::sync::Api::new()?;

        println!("  Downloading {} from {} ...", DEFAULT_GGUF_FILE, DEFAULT_GGUF_REPO);
        let gguf = api.model(DEFAULT_GGUF_REPO.to_string()).get(DEFAULT_GGUF_FILE)?;
        println!("  GGUF: {}", gguf.display());

        println!("  Downloading tokenizer.json from {} ...", DEFAULT_TOKENIZER_REPO);
        let tok = api.model(DEFAULT_TOKENIZER_REPO.to_string()).get("tokenizer.json")?;
        println!("  Tokenizer: {}", tok.display());

        (gguf, Some(tok))
    };

    // ── Step 1: Load GGUF file ───────────────────────────────────────
    println!("\nLoading GGUF: {}", gguf_path.display());
    let (config, weights) = ModelLoader::load_gguf(&gguf_path)?;

    // ── Print quantization stats ─────────────────────────────────────
    print_quant_stats(&weights);

    // ── Step 2: Build model ──────────────────────────────────────────
    println!("\nBuilding model...");
    let model = LlmModel::from_pretrained(&config, weights)?;

    // ── Step 3: Load tokenizer ───────────────────────────────────────
    let tokenizer: Box<dyn Tokenizer> = if let Some(tok_path) = &tokenizer_path {
        println!("Loading tokenizer: {}", tok_path.display());
        Box::new(HFTokenizer::from_file(tok_path)?)
    } else {
        println!("No tokenizer provided — using NaiveTokenizer (byte-level)");
        Box::new(NaiveTokenizer::new())
    };

    // ── Print model stats ────────────────────────────────────────────
    let (total_params, _) = model.parameter_count();
    println!("\n--- Model Stats ---");
    println!("  Parameters : {:.1}M", total_params as f64 / 1e6);
    println!("  Dimension  : {}", config.dim);
    println!("  Layers     : {}", config.n_layers);
    println!("  Heads      : {}", config.n_heads);
    println!("  KV Heads   : {}", config.n_kv_heads.unwrap_or(config.n_heads));
    println!("  Hidden dim : {}", config.hidden_dim);
    println!("  Vocab      : {}", config.vocab_size);
    println!("  Max seq len: {}", config.max_seq_len);
    println!("  Position   : {:?}", config.position_encoding);

    // ── Step 4: Build generator ──────────────────────────────────────
    let generator = Generator::new(&model, tokenizer.as_ref(), temperature)
        .with_top_k(DEFAULT_TOP_K)
        .with_top_p(DEFAULT_TOP_P);

    // ── Step 5: Interactive REPL ─────────────────────────────────────
    println!("\n--- Interactive Generation ---");
    println!("  Temperature: {temperature}  Top-k: {DEFAULT_TOP_K}  Top-p: {DEFAULT_TOP_P}  Max tokens: {max_tokens}");
    println!("  Type a prompt and press Enter. Type 'quit' to exit.\n");

    loop {
        print!("Prompt> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let prompt = input.trim();

        if prompt.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }
        if prompt.is_empty() {
            continue;
        }

        print!("\n");
        io::stdout().flush()?;

        let start = std::time::Instant::now();
        let mut token_count = 0u32;

        let _output = generator.generate_stream(prompt, max_tokens, |token_id| {
            token_count += 1;
            if let Ok(piece) = tokenizer.decode(&[token_id]) {
                print!("{}", piece);
                let _ = io::stdout().flush();
            }
            true
        })?;

        let elapsed = start.elapsed();
        let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
            token_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        println!(
            "\n\n[{} tokens in {:.2}s — {:.1} tok/s]\n",
            token_count,
            elapsed.as_secs_f64(),
            tok_per_sec
        );
    }

    Ok(())
}

/// Print a breakdown of tensor dtypes (quantization levels) in the loaded weights.
fn print_quant_stats(weights: &HashMap<String, llmforge::Tensor>) {
    let mut f32_count = 0usize;
    let mut f16_count = 0usize;
    let mut q8_count = 0usize;
    let mut q4_count = 0usize;
    let mut other_count = 0usize;

    for tensor in weights.values() {
        match tensor.dtype() {
            DType::F32 => f32_count += 1,
            DType::F16 => f16_count += 1,
            DType::Q8_0 => q8_count += 1,
            DType::Q4_0 => q4_count += 1,
            _ => other_count += 1,
        }
    }

    let total = weights.len();
    println!("\n--- Quantization Breakdown ({} tensors) ---", total);
    if q4_count > 0 {
        println!("  Q4_0 : {} tensors", q4_count);
    }
    if q8_count > 0 {
        println!("  Q8_0 : {} tensors", q8_count);
    }
    if f16_count > 0 {
        println!("  F16  : {} tensors", f16_count);
    }
    if f32_count > 0 {
        println!("  F32  : {} tensors", f32_count);
    }
    if other_count > 0 {
        println!("  Other: {} tensors", other_count);
    }
}
