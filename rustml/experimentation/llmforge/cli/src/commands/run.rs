use std::collections::HashMap;
use std::io::{self, Write};

use anyhow::{bail, Context, Result};

use llmforge::config::ModelConfig;
use llmforge::inference::Generator;
use llmforge::loader::{ModelLoader, WeightMap};
use llmforge::models::LlmModel;
use llmforge::tokenization::{HFTokenizer, NaiveTokenizer, Tokenizer};
use llmforge::{RuntimeConfig, Tensor};

use crate::args::{Arch, RunArgs};
use crate::model_source::{self, ModelFormat, ResolvedModel};

pub fn execute(args: &RunArgs) -> Result<()> {
    // Apply threading config
    RuntimeConfig {
        num_threads: args.threads,
    }
    .apply()
    .context("Failed to apply runtime config")?;

    // Resolve model source
    let resolved = model_source::resolve(
        &args.model,
        args.file.as_deref(),
        args.config.as_deref(),
        args.tokenizer.as_deref(),
    )?;

    eprintln!("Model file: {}", resolved.model_path.display());
    if let Some(ref c) = resolved.config_path {
        eprintln!("Config:     {}", c.display());
    }
    if let Some(ref t) = resolved.tokenizer_path {
        eprintln!("Tokenizer:  {}", t.display());
    }

    // Load model
    let (config, model) = match resolved.format {
        ModelFormat::Gguf => load_gguf(&resolved)?,
        ModelFormat::SafeTensors => load_safetensors(&resolved, args)?,
    };

    // Load tokenizer:
    //  1. Explicit tokenizer.json (from --tokenizer, sibling file, or HF repo)
    //  2. GGUF embedded → generate tokenizer.json, then load via HFTokenizer
    //  3. GGUF embedded → in-memory GGUFTokenizer (fallback if generation fails)
    //  4. NaiveTokenizer (last resort)
    let mut tok_path = resolved.tokenizer_path.clone();

    // For GGUF models without a tokenizer.json, try to generate one from metadata
    if tok_path.is_none() && resolved.format == ModelFormat::Gguf {
        eprintln!("No tokenizer.json found — checking GGUF metadata...");
        match crate::gguf_tokenizer::ensure_tokenizer_json(&resolved.model_path) {
            Ok(Some(path)) => {
                eprintln!("Tokenizer:  {}", path.display());
                tok_path = Some(path);
            }
            Ok(None) => {
                eprintln!("  GGUF file has no embedded tokenizer metadata");
            }
            Err(e) => {
                eprintln!("  Warning: failed to generate tokenizer.json: {}", e);
            }
        }
    }

    let tokenizer: Box<dyn Tokenizer> = if let Some(ref path) = tok_path {
        eprintln!("Loading tokenizer...");
        Box::new(
            HFTokenizer::from_file(path)
                .with_context(|| format!("Failed to load tokenizer from {}", path.display()))?,
        )
    } else if resolved.format == ModelFormat::Gguf {
        // Fallback: in-memory tokenizer from GGUF metadata
        eprintln!("Extracting tokenizer from GGUF metadata (in-memory)...");
        match crate::gguf_tokenizer::GGUFTokenizer::from_gguf_file(&resolved.model_path) {
            Ok(tok) => Box::new(tok),
            Err(e) => {
                eprintln!("  Warning: failed to extract GGUF tokenizer: {}", e);
                eprintln!("  Falling back to byte-level NaiveTokenizer");
                Box::new(NaiveTokenizer::new())
            }
        }
    } else {
        eprintln!("No tokenizer found — using byte-level NaiveTokenizer");
        Box::new(NaiveTokenizer::new())
    };

    // Print stats
    let (total_params, _) = model.parameter_count();
    eprintln!();
    eprintln!("--- Model Stats ---");
    eprintln!("  Parameters : {:.1}M", total_params as f64 / 1e6);
    eprintln!("  Dimension  : {}", config.dim);
    eprintln!("  Layers     : {}", config.n_layers);
    eprintln!("  Heads      : {}", config.n_heads);
    eprintln!("  KV Heads   : {:?}", config.n_kv_heads);
    eprintln!("  Hidden dim : {}", config.hidden_dim);
    eprintln!("  Vocab      : {}", config.vocab_size);
    eprintln!("  Max seq len: {}", config.max_seq_len);
    eprintln!("  Pos enc    : {:?}", config.position_encoding);

    // Build generator
    let mut gen = Generator::new(model.as_ref(), tokenizer.as_ref(), args.temperature)
        .with_top_k(args.top_k)
        .with_top_p(args.top_p);

    if let Some(bos) = config.bos_token_id {
        eprintln!("  BOS token  : {}", bos);
        gen = gen.with_bos_token(bos);
    }
    if let Some(eos) = config.eos_token_id {
        eprintln!("  EOS token  : {}", eos);
        gen = gen.with_eos_token(eos);
    }

    if args.repetition_penalty != 1.0 {
        gen = gen.with_repetition_penalty(args.repetition_penalty);
    }

    // Decide mode
    if let Some(ref prompt) = args.prompt {
        if args.interactive {
            // Single-shot first, then REPL
            run_single_shot(&gen, &*tokenizer, prompt, args.max_tokens)?;
            run_repl(&gen, &*tokenizer, args)?;
        } else {
            run_single_shot(&gen, &*tokenizer, prompt, args.max_tokens)?;
        }
    } else {
        run_repl(&gen, &*tokenizer, args)?;
    }

    Ok(())
}

fn load_gguf(resolved: &ResolvedModel) -> Result<(ModelConfig, Box<LlmModel>)> {
    eprintln!("Loading GGUF...");
    let (config, weights) = ModelLoader::load_gguf(&resolved.model_path)
        .context("Failed to load GGUF file")?;

    print_quant_stats(&weights);

    eprintln!("Building model...");
    let model = LlmModel::from_pretrained(&config, weights)
        .context("Failed to build model from GGUF weights")?;

    Ok((config, Box::new(model)))
}

fn load_safetensors(resolved: &ResolvedModel, args: &RunArgs) -> Result<(ModelConfig, Box<LlmModel>)> {
    let config_path = resolved
        .config_path
        .as_ref()
        .context("SafeTensors model requires config.json (use --config or place alongside model)")?;

    // Detect architecture
    let arch = match args.arch {
        Arch::Gpt2 => "gpt2",
        Arch::Llama => "llama",
        Arch::Auto => model_source::detect_arch(config_path)?,
    };
    eprintln!("Architecture: {}", arch);

    // Parse config
    let config = match arch {
        "gpt2" => ModelConfig::from_hf_gpt2(config_path)
            .context("Failed to parse GPT-2 config")?,
        "llama" => ModelConfig::from_hf_llama2(config_path)
            .context("Failed to parse Llama config")?,
        _ => bail!("Unknown architecture: {}", arch),
    };

    // Load weights
    eprintln!("Loading SafeTensors weights...");
    let raw_weights = ModelLoader::load_safetensors(&resolved.model_path)
        .context("Failed to load SafeTensors file")?;
    eprintln!("  Loaded {} raw tensors", raw_weights.len());

    // Remap
    eprintln!("Remapping weights...");
    let (weight_map, model) = match arch {
        "gpt2" => {
            let wm = WeightMap::gpt2(config.n_layers);
            let weights = wm.remap(raw_weights)?;
            let m = LlmModel::from_pretrained_gpt2(&config, weights)
                .context("Failed to build GPT-2 model")?;
            (wm, m)
        }
        "llama" => {
            let wm = WeightMap::llama2(config.n_layers);
            let weights = wm.remap(raw_weights)?;
            let m = LlmModel::from_pretrained(&config, weights)
                .context("Failed to build Llama model")?;
            (wm, m)
        }
        _ => unreachable!(),
    };
    let _ = weight_map; // used only for remap

    Ok((config, Box::new(model)))
}

fn run_single_shot(
    gen: &Generator,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    max_tokens: usize,
) -> Result<()> {
    let start = std::time::Instant::now();
    let mut token_count = 0u32;

    // Incremental decode: maintain buffer of all generated token IDs and diff
    // the decoded text to get the new piece. This correctly handles SentencePiece
    // tokenizers where single-token decode strips the leading space marker (▁).
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut prev_text = String::new();

    let _output = gen
        .generate_stream(prompt, max_tokens, |token_id| {
            token_count += 1;
            all_tokens.push(token_id);
            if let Ok(full_text) = tokenizer.decode(&all_tokens) {
                let new_text = &full_text[prev_text.len()..];
                print!("{}", new_text);
                let _ = io::stdout().flush();
                prev_text = full_text;
            }
            true
        })
        .context("Generation failed")?;

    let elapsed = start.elapsed();
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        token_count as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    eprintln!(
        "\n[{} tokens in {:.2}s — {:.1} tok/s]",
        token_count,
        elapsed.as_secs_f64(),
        tok_per_sec
    );

    Ok(())
}

fn run_repl(gen: &Generator, tokenizer: &dyn Tokenizer, args: &RunArgs) -> Result<()> {
    eprintln!();
    eprintln!("--- Interactive Generation ---");
    eprintln!(
        "  Temperature: {}  Top-k: {}  Top-p: {}  Max tokens: {}",
        args.temperature, args.top_k, args.top_p, args.max_tokens
    );
    eprintln!("  Type a prompt and press Enter. Type 'quit' to exit.");
    eprintln!();

    loop {
        eprint!("Prompt> ");
        io::stderr().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let prompt = input.trim();

        if prompt.eq_ignore_ascii_case("quit") {
            eprintln!("Goodbye!");
            break;
        }
        if prompt.is_empty() {
            continue;
        }

        println!();
        run_single_shot(gen, tokenizer, prompt, args.max_tokens)?;
        println!();
    }

    Ok(())
}

fn print_quant_stats(weights: &HashMap<String, Tensor>) {
    use llmforge::core::tensor::DType;

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
    eprintln!();
    eprintln!("--- Quantization Breakdown ({} tensors) ---", total);
    if q4_count > 0 {
        eprintln!("  Q4_0 : {} tensors", q4_count);
    }
    if q8_count > 0 {
        eprintln!("  Q8_0 : {} tensors", q8_count);
    }
    if f16_count > 0 {
        eprintln!("  F16  : {} tensors", f16_count);
    }
    if f32_count > 0 {
        eprintln!("  F32  : {} tensors", f32_count);
    }
    if other_count > 0 {
        eprintln!("  Other: {} tensors", other_count);
    }
}
