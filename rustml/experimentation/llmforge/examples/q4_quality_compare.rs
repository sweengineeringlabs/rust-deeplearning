/// A/B comparison: dequant-to-F32 vs native Q4_0×Q8_0 integer matmul.
///
/// Loads TinyLlama-1.1B Q4_0, runs the same prompts through both paths
/// at temperature=0 (greedy, deterministic), and prints outputs side-by-side.
///
/// Usage:
///   cargo run --release --example q4_quality_compare
///   cargo run --release --example q4_quality_compare -- --max-tokens 64

use llmforge::inference::Generator;
use llmforge::loader::ModelLoader;
use llmforge::models::LlmModel;
use llmforge::tokenization::{HFTokenizer, Tokenizer};

use std::io::{self, Write};

const DEFAULT_GGUF_REPO: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
const DEFAULT_GGUF_FILE: &str = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
const DEFAULT_TOKENIZER_REPO: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

fn generate_greedy(
    model: &LlmModel,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    max_tokens: usize,
    bos: Option<u32>,
    eos: Option<u32>,
) -> Result<(String, f64), Box<dyn std::error::Error>> {
    generate_greedy_with_template(model, tokenizer, prompt, max_tokens, None, bos, eos)
}

fn generate_greedy_with_template(
    model: &LlmModel,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    max_tokens: usize,
    chat_template: Option<String>,
    bos: Option<u32>,
    eos: Option<u32>,
) -> Result<(String, f64), Box<dyn std::error::Error>> {
    let mut generator = Generator::new(model, tokenizer, 0.0) // temp=0 → greedy
        .with_chat_template(chat_template);
    if let Some(b) = bos { generator = generator.with_bos_token(b); }
    if let Some(e) = eos { generator = generator.with_eos_token(e); }

    let start = std::time::Instant::now();
    let mut tokens = Vec::new();
    let _output = generator.generate_stream(prompt, max_tokens, |token_id| {
        tokens.push(token_id);
        true
    })?;

    let elapsed = start.elapsed().as_secs_f64();
    let tok_per_sec = if elapsed > 0.0 { tokens.len() as f64 / elapsed } else { 0.0 };

    // Decode full output
    let text = tokenizer.decode(&tokens).unwrap_or_else(|_| format!("[{} tokens]", tokens.len()));
    Ok((text, tok_per_sec))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut max_tokens: usize = 128;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--max-tokens" {
            i += 1;
            max_tokens = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(max_tokens);
        }
        i += 1;
    }

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  A/B Generation Quality: dequant-F32 vs native Q4×Q8 i32   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // ── Download / load model ────────────────────────────────────────
    let api = hf_hub::api::sync::Api::new()?;

    print!("Downloading model...  ");
    io::stdout().flush()?;
    let gguf_path = api.model(DEFAULT_GGUF_REPO.to_string()).get(DEFAULT_GGUF_FILE)?;
    println!("done.");

    print!("Downloading tokenizer...  ");
    io::stdout().flush()?;
    let tok_path = api.model(DEFAULT_TOKENIZER_REPO.to_string()).get("tokenizer.json")?;
    println!("done.");

    print!("Loading GGUF...  ");
    io::stdout().flush()?;
    let (config, weights) = ModelLoader::load_gguf(&gguf_path)?;
    println!("done.");

    print!("Building model...  ");
    io::stdout().flush()?;
    let mut model = LlmModel::from_pretrained(&config, weights)?;
    println!("done.");

    let tokenizer = HFTokenizer::from_file(&tok_path)?;

    let bos = config.bos_token_id;
    let eos = config.eos_token_id;

    let (total_params, _) = model.parameter_count();
    println!("\nModel: TinyLlama-1.1B Q4_0 ({:.1}M params)", total_params as f64 / 1e6);
    println!("Settings: temperature=0 (greedy), max_tokens={}", max_tokens);
    println!("BOS={:?}, EOS={:?}\n", bos, eos);

    // ── Test prompts ─────────────────────────────────────────────────
    let prompts = [
        // Factual
        ("Factual", "The capital of France is"),
        ("Knowledge", "The largest planet in our solar system is"),
        // Instruction-following
        ("Instruction", "List three primary colors:\n1."),
        // Open-ended / creative
        ("Creative", "Once upon a time in a magical forest, there lived"),
        ("Story", "The old lighthouse keeper watched the storm approach and"),
    ];

    // ── Run both paths ───────────────────────────────────────────────
    let separator = "─".repeat(72);

    for (label, prompt) in &prompts {
        println!("{}", separator);
        println!("[{}] Prompt: \"{}\"", label, prompt);
        println!("{}", separator);

        // Path A: dequant-to-F32 (default)
        model.set_native_q4_matmul(false);
        let (text_a, tps_a) = generate_greedy(&model, &tokenizer, prompt, max_tokens, bos, eos)?;

        // Path B: native integer
        model.set_native_q4_matmul(true);
        let (text_b, tps_b) = generate_greedy(&model, &tokenizer, prompt, max_tokens, bos, eos)?;

        println!("\n  A (dequant F32) [{:.1} tok/s]:", tps_a);
        for line in text_a.lines() {
            println!("    {}", line);
        }

        println!("\n  B (native i32) [{:.1} tok/s]:", tps_b);
        for line in text_b.lines() {
            println!("    {}", line);
        }

        // Quick comparison
        let same = text_a == text_b;
        if same {
            println!("\n  Result: IDENTICAL");
        } else {
            // Count matching tokens at the start
            let tokens_a: Vec<&str> = text_a.split_whitespace().collect();
            let tokens_b: Vec<&str> = text_b.split_whitespace().collect();
            let matching = tokens_a.iter().zip(&tokens_b).take_while(|(a, b)| a == b).count();
            let _total = tokens_a.len().max(tokens_b.len());
            println!(
                "\n  Result: DIVERGED after ~{} words ({} vs {} total words)",
                matching,
                tokens_a.len(),
                tokens_b.len()
            );
        }
        println!();
    }

    // ── Chat template comparison ────────────────────────────────────
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Chat Template: raw prompt vs wrapped prompt                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    model.set_native_q4_matmul(false);

    let chat_prompts = [
        ("Factual", "The capital of France is"),
        ("Knowledge", "The largest planet in our solar system is"),
        ("Instruction", "List three primary colors"),
    ];

    // Use the template from GGUF metadata if available, otherwise hardcode TinyLlama's
    let template = config.chat_template.clone().unwrap_or_else(|| {
        "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{{ prompt }}</s>\n<|assistant|>\n".to_string()
    });

    for (label, prompt) in &chat_prompts {
        println!("{}", separator);
        println!("[{}] Prompt: \"{}\"", label, prompt);
        println!("{}", separator);

        // Raw (no template)
        let (text_raw, tps_raw) = generate_greedy(&model, &tokenizer, prompt, max_tokens, bos, eos)?;

        // Wrapped (with chat template)
        let (text_wrapped, tps_wrapped) = generate_greedy_with_template(
            &model, &tokenizer, prompt, max_tokens, Some(template.clone()), bos, eos,
        )?;

        println!("\n  Raw (no template) [{:.1} tok/s]:", tps_raw);
        for line in text_raw.lines().take(5) {
            println!("    {}", line);
        }

        println!("\n  Wrapped (chat template) [{:.1} tok/s]:", tps_wrapped);
        for line in text_wrapped.lines().take(5) {
            println!("    {}", line);
        }
        println!();
    }

    println!("{}", separator);
    println!("Done.");
    Ok(())
}
