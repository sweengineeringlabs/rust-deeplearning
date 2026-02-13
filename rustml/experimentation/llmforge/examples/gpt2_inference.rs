//! GPT-2 (124M) end-to-end inference example.
//!
//! Auto-downloads model files from HuggingFace (`openai-community/gpt2`),
//! loads weights via SafeTensors, and runs an interactive generation REPL.
//!
//! ```bash
//! cargo run --example gpt2_inference
//! ```

use llmforge::config::ModelConfig;
use llmforge::inference::Generator;
use llmforge::loader::{ModelLoader, WeightMap};
use llmforge::models::LlmModel;
use llmforge::tokenization::{HFTokenizer, Tokenizer};

use std::io::{self, Write};

// ── Generation defaults ──────────────────────────────────────────────
const TEMPERATURE: f32 = 0.8;
const TOP_K: usize = 40;
const TOP_P: f32 = 0.95;
const MAX_TOKENS: usize = 128;
const GPT2_EOS_TOKEN: u32 = 50256; // <|endoftext|>

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LLMForge: GPT-2 (124M) Inference ===\n");

    // ── Step 1: Download model files from HuggingFace ────────────────
    println!("Downloading model from openai-community/gpt2 ...");
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("openai-community/gpt2".to_string());

    let config_path = repo.get("config.json")?;
    let model_path = repo.get("model.safetensors")?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    println!("  config.json    : {}", config_path.display());
    println!("  model.safetensors: {}", model_path.display());
    println!("  tokenizer.json : {}", tokenizer_path.display());

    // ── Step 2: Parse config ─────────────────────────────────────────
    println!("\nParsing config...");
    let config = ModelConfig::from_hf_gpt2(&config_path)?;

    // ── Step 3: Load SafeTensors weights ─────────────────────────────
    println!("Loading SafeTensors weights...");
    let raw_weights = ModelLoader::load_safetensors(&model_path)?;
    println!("  Loaded {} raw tensors", raw_weights.len());

    // ── Step 4: Remap weight names ───────────────────────────────────
    println!("Remapping weights (HuggingFace → LLMForge)...");
    let weight_map = WeightMap::gpt2(config.n_layers);
    let weights = weight_map.remap(raw_weights)?;
    println!("  Mapped {} tensors", weights.len());

    // ── Step 5: Build model ──────────────────────────────────────────
    println!("Building GPT-2 model...");
    let model = LlmModel::from_pretrained_gpt2(&config, weights)?;

    // ── Step 6: Load tokenizer ───────────────────────────────────────
    println!("Loading tokenizer...");
    let tokenizer = HFTokenizer::from_file(&tokenizer_path)?;

    // ── Print model stats ────────────────────────────────────────────
    let (total_params, _) = model.parameter_count();
    println!("\n--- Model Stats ---");
    println!("  Parameters : {:.1}M", total_params as f64 / 1e6);
    println!("  Dimension  : {}", config.dim);
    println!("  Layers     : {}", config.n_layers);
    println!("  Heads      : {}", config.n_heads);
    println!("  Vocab      : {}", tokenizer.vocab_size());
    println!("  Max seq len: {}", config.max_seq_len);

    // ── Step 7: Build generator ──────────────────────────────────────
    let generator = Generator::new(&model, &tokenizer, TEMPERATURE)
        .with_top_k(TOP_K)
        .with_top_p(TOP_P)
        .with_eos_token(GPT2_EOS_TOKEN);

    // ── Step 8: Interactive REPL ─────────────────────────────────────
    println!("\n--- Interactive Generation ---");
    println!("  Temperature: {TEMPERATURE}  Top-k: {TOP_K}  Top-p: {TOP_P}  Max tokens: {MAX_TOKENS}");
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

        let _output = generator.generate_stream(prompt, MAX_TOKENS, |token_id| {
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

        println!("\n\n[{} tokens in {:.2}s — {:.1} tok/s]\n", token_count, elapsed.as_secs_f64(), tok_per_sec);
    }

    Ok(())
}
