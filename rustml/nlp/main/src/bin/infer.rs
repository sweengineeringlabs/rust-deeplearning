use std::io::{self, Read};
use std::path::PathBuf;

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
    #[arg(long)]
    prompt: Option<String>,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value_t = 0.8)]
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

    // 1. Parse GGUF header
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

    // 7. Validate sampling parameters
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

    // 8. Read prompt and generate
    let prompt = read_prompt(&cli)?;
    eprintln!("---");

    if cli.stream {
        let _output = generator.generate_stream(&prompt, cli.max_tokens, |token_id| {
            match tokenizer.decode(&[token_id]) {
                Ok(piece) => print!("{piece}"),
                Err(e) => eprintln!("[warn] failed to decode token {}: {}", token_id, e),
            }
            true
        })?;
        println!();
    } else {
        let output = generator.generate(&prompt, cli.max_tokens)?;
        println!("{output}");
    }

    Ok(())
}
