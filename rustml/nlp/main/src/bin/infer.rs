use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;

use rustml_gguf::GGUFFile;
use rustml_nlp::{Generator, LlmModel, convert_tensors, gguf_config_to_model_config};
use rustml_tokenizer::{GgufTokenizer, Tokenizer};

/// RustML Inference CLI — run text generation on a GGUF model.
#[derive(Parser)]
#[command(name = "rustml-infer", version, about)]
struct Cli {
    /// Path to a GGUF model file.
    gguf_path: PathBuf,

    /// Prompt text. Reads from stdin if omitted.
    #[arg(long, conflicts_with = "batch_file")]
    prompt: Option<String>,

    /// File with one prompt per line. Runs parallel generation via rayon.
    #[arg(long)]
    batch_file: Option<PathBuf>,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value_t = 0.8, allow_negative_numbers = true)]
    temperature: f32,

    /// Top-k sampling.
    #[arg(long)]
    top_k: Option<usize>,

    /// Nucleus (top-p) sampling.
    #[arg(long)]
    top_p: Option<f32>,

    /// Repetition penalty.
    #[arg(long)]
    repetition_penalty: Option<f32>,

    /// Print tokens as they are generated.
    #[arg(long)]
    stream: bool,

    /// Wrap the prompt in a chat template extracted from GGUF metadata.
    #[arg(long)]
    chat: bool,

    /// Generation timeout in seconds. Stops generation if exceeded.
    #[arg(long, allow_negative_numbers = true)]
    timeout: Option<f64>,
}

fn read_prompt(cli: &Cli) -> Result<String> {
    if let Some(ref text) = cli.prompt {
        Ok(text.clone())
    } else {
        eprintln!("Reading prompt from stdin...");
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .context("Failed to read stdin")?;
        if buf.trim().is_empty() {
            bail!("No prompt provided (use --prompt or pipe text to stdin)");
        }
        Ok(buf)
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // 1. Validate parameters (fail fast before loading model)
    if cli.temperature < 0.0 {
        bail!("--temperature must be >= 0.0, got {}", cli.temperature);
    }
    if let Some(k) = cli.top_k {
        if k == 0 {
            bail!("--top-k must be > 0");
        }
    }
    if let Some(p) = cli.top_p {
        if p <= 0.0 || p > 1.0 {
            bail!("--top-p must be in (0.0, 1.0], got {}", p);
        }
    }
    if let Some(rp) = cli.repetition_penalty {
        if rp <= 0.0 {
            bail!("--repetition-penalty must be > 0.0, got {}", rp);
        }
    }
    if let Some(secs) = cli.timeout {
        if secs <= 0.0 {
            bail!("--timeout must be > 0.0, got {}", secs);
        }
    }
    if cli.stream && cli.batch_file.is_some() {
        bail!("--stream is not supported with --batch-file");
    }
    let batch_contents = if let Some(ref batch_path) = cli.batch_file {
        let contents = fs::read_to_string(batch_path)
            .with_context(|| format!("Failed to read batch file: {}", batch_path.display()))?;
        if contents.lines().all(|l| l.trim().is_empty()) {
            bail!("Batch file is empty: {}", batch_path.display());
        }
        Some(contents)
    } else {
        None
    };

    // 2. Parse GGUF header
    eprintln!("Loading GGUF: {}", cli.gguf_path.display());
    let gguf = GGUFFile::parse_header(&cli.gguf_path)
        .with_context(|| format!("Failed to parse GGUF: {}", cli.gguf_path.display()))?;
    eprintln!(
        "  GGUF v{}, {} tensors",
        gguf.version,
        gguf.tensor_infos.len()
    );

    // 2. Extract config
    let gguf_config = gguf
        .to_model_config()
        .with_context(|| "Failed to extract model config from GGUF")?;
    let config = gguf_config_to_model_config(&gguf_config)
        .with_context(|| "Failed to convert GGUF config to model config")?;
    eprintln!(
        "  arch={}, dim={}, layers={}, heads={}, vocab={}",
        gguf_config.architecture, config.dim, config.n_layers, config.n_heads, config.vocab_size
    );

    // 3. Build tokenizer
    let tokenizer = GgufTokenizer::from_gguf(&gguf)
        .with_context(|| "Failed to build tokenizer from GGUF")?;
    eprintln!("  Tokenizer: {} tokens", tokenizer.vocab_size());

    // 4. Load tensors (architecture-dependent)
    eprintln!("  Loading tensors...");
    let is_gemma3 = gguf_config.architecture == "gemma3";
    let loaded_tensors = if is_gemma3 {
        gguf.load_and_remap_gemma3(&cli.gguf_path, config.n_layers)
            .with_context(|| "Failed to load/remap gemma3 tensors")?
    } else {
        gguf.load_and_remap(&cli.gguf_path, config.n_layers)
            .with_context(|| "Failed to load/remap tensors")?
    };
    let tensors = convert_tensors(loaded_tensors);
    eprintln!("  {} tensors loaded", tensors.len());

    // 5. Build model
    eprintln!("  Building model...");
    let model = if is_gemma3 {
        LlmModel::from_pretrained_gemma3(&config, tensors)
            .with_context(|| "Failed to build gemma3 model")?
    } else {
        LlmModel::from_pretrained(&config, tensors)
            .with_context(|| "Failed to build model")?
    };
    let (total_params, _) = model.parameter_count();
    eprintln!("  Model ready: {:.1}M params", total_params as f64 / 1e6);

    // Report KV cache memory
    let n_kv_heads = config.n_kv_heads.unwrap_or(config.n_heads);
    let head_dim = config.head_dim.unwrap_or(config.dim / config.n_heads);
    let cache_bytes = 2 * config.n_layers * 1 * n_kv_heads * config.max_seq_len * head_dim * 4;
    let cache_mb = cache_bytes as f64 / (1024.0 * 1024.0);
    eprintln!("  KV cache: {:.1} MB ({}layers x {}heads x {}seq x {}dim x f32 x 2)",
        cache_mb, config.n_layers, n_kv_heads, config.max_seq_len, head_dim);

    // 6. Build generator
    let mut generator = Generator::new(&model, &tokenizer, cli.temperature);

    if let Some(k) = cli.top_k {
        generator = generator.with_top_k(k);
    }
    if let Some(p) = cli.top_p {
        generator = generator.with_top_p(p);
    }
    if let Some(rp) = cli.repetition_penalty {
        generator = generator.with_repetition_penalty(rp);
    }
    if let Some(eos) = config.eos_token_id {
        generator = generator.with_eos_token(eos);
    }
    if let Some(bos) = config.bos_token_id {
        generator = generator.with_bos_token(bos);
    }
    if cli.chat {
        generator = generator.with_chat_template(config.chat_template.clone());
    }
    if let Some(secs) = cli.timeout {
        generator = generator.with_timeout(Duration::from_secs_f64(secs));
        eprintln!("  Timeout: {:.1}s", secs);
    }

    // 8. Read prompt(s) and generate
    if let Some(ref contents) = batch_contents {
        let prompts: Vec<&str> = contents.lines().filter(|l| !l.trim().is_empty()).collect();
        eprintln!("  Batch: {} prompts", prompts.len());
        eprintln!("---");

        let gen_start = Instant::now();
        let results = generator.generate_batch_parallel(&prompts, cli.max_tokens)?;
        let elapsed = gen_start.elapsed();

        for (i, output) in results.iter().enumerate() {
            println!("[{}] {}", i, output);
        }
        eprintln!("---");
        eprintln!("  {} prompts in {:.2}s ({:.1} prompts/sec)",
            results.len(), elapsed.as_secs_f64(),
            results.len() as f64 / elapsed.as_secs_f64().max(1e-9));
    } else {
        let prompt = read_prompt(&cli)?;
        eprintln!("---");

        let gen_start = Instant::now();

        if cli.stream {
            let mut token_count: usize = 0;
            let _output = generator.generate_stream(&prompt, cli.max_tokens, |token_id| {
                token_count += 1;
                match tokenizer.decode(&[token_id]) {
                    Ok(piece) => print!("{piece}"),
                    Err(e) => eprintln!("[warn] failed to decode token {}: {}", token_id, e),
                }
                true
            })?;
            println!();
            let elapsed = gen_start.elapsed();
            let tps = if elapsed.as_secs_f64() > 0.0 {
                token_count as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            eprintln!("---");
            eprintln!("  {} tokens in {:.2}s ({:.1} tokens/sec)", token_count, elapsed.as_secs_f64(), tps);
        } else {
            let output = generator.generate(&prompt, cli.max_tokens)?;
            let elapsed = gen_start.elapsed();
            println!("{output}");
            eprintln!("---");
            eprintln!("  Generated in {:.2}s", elapsed.as_secs_f64());
        }
    }

    Ok(())
}
