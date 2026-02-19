use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Args;

use rustml_gguf::GGUFFile;
use rustml_hub::HubApi;
use rustml_nlp::{
    Generator, LanguageModel, LlmModel, OptProfile, convert_tensors,
    gguf_config_to_model_config,
};
use rustml_tokenizer::{BpeTokenizer, GgufTokenizer, HFTokenizer, Tokenizer};

#[derive(Args)]
pub struct InferArgs {
    /// Path to a GGUF model file.
    #[arg(conflicts_with = "safetensors")]
    gguf_path: Option<PathBuf>,

    /// HuggingFace model ID to load via SafeTensors (e.g. openai-community/gpt2).
    #[arg(long)]
    safetensors: Option<String>,

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

    /// Start an interactive multi-turn chat session (implies --chat --stream).
    #[arg(long, conflicts_with_all = ["prompt", "batch_file"])]
    interactive: bool,

    /// Wrap the prompt in a chat template extracted from GGUF metadata.
    #[arg(long)]
    chat: bool,

    /// Generation timeout in seconds. Stops generation if exceeded.
    #[arg(long, allow_negative_numbers = true)]
    timeout: Option<f64>,

    /// KV cache context length. Auto-sized from prompt + max_tokens if omitted.
    #[arg(long)]
    context_len: Option<usize>,

    /// Optimization profile: optimized (default), baseline (all opts off), aggressive (lower thresholds).
    #[arg(long, default_value = "optimized")]
    opt_profile: String,
}

fn read_prompt(args: &InferArgs) -> Result<String> {
    if let Some(ref text) = args.prompt {
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

fn validate_args(args: &InferArgs) -> Result<Option<String>> {
    if args.gguf_path.is_none() && args.safetensors.is_none() {
        bail!("Provide a GGUF model path or --safetensors <MODEL_ID>");
    }
    if args.temperature < 0.0 {
        bail!("--temperature must be >= 0.0, got {}", args.temperature);
    }
    if let Some(k) = args.top_k {
        if k == 0 {
            bail!("--top-k must be > 0");
        }
    }
    if let Some(p) = args.top_p {
        if p <= 0.0 || p > 1.0 {
            bail!("--top-p must be in (0.0, 1.0], got {}", p);
        }
    }
    if let Some(rp) = args.repetition_penalty {
        if rp <= 0.0 {
            bail!("--repetition-penalty must be > 0.0, got {}", rp);
        }
    }
    if let Some(secs) = args.timeout {
        if secs <= 0.0 {
            bail!("--timeout must be > 0.0, got {}", secs);
        }
    }
    if let Some(cl) = args.context_len {
        if cl == 0 {
            bail!("--context-len must be > 0");
        }
    }
    if args.stream && args.batch_file.is_some() {
        bail!("--stream is not supported with --batch-file");
    }

    let batch_contents = if let Some(ref batch_path) = args.batch_file {
        let contents = fs::read_to_string(batch_path)
            .with_context(|| format!("Failed to read batch file: {}", batch_path.display()))?;
        if contents.lines().all(|l| l.trim().is_empty()) {
            bail!("Batch file is empty: {}", batch_path.display());
        }
        Some(contents)
    } else {
        None
    };

    Ok(batch_contents)
}

fn parse_opt_profile(s: &str) -> Result<OptProfile> {
    match s {
        "optimized" => Ok(OptProfile::Optimized),
        "baseline" => Ok(OptProfile::Baseline),
        "aggressive" => Ok(OptProfile::Aggressive),
        other => bail!("Unknown --opt-profile '{}' (expected: optimized, baseline, aggressive)", other),
    }
}

fn run_generation(
    model: &(dyn LanguageModel + Sync),
    tokenizer: &(dyn Tokenizer + Sync),
    args: &InferArgs,
    batch_contents: Option<String>,
    eos_token_id: Option<u32>,
    bos_token_id: Option<u32>,
    chat_template: Option<String>,
    profile: OptProfile,
) -> Result<()> {
    // Interactive mode auto-enables chat.
    let use_chat = args.chat || args.interactive;
    if args.interactive && chat_template.is_none() {
        eprintln!("  [warn] no chat template found in model; interactive mode may produce poor results");
    }

    // Read prompt(s) early so we can compute effective KV cache context length.
    // Interactive mode doesn't need prompts upfront.
    let (prompts, is_batch) = if args.interactive {
        (vec![], false)
    } else if let Some(ref contents) = batch_contents {
        let ps: Vec<String> = contents
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(String::from)
            .collect();
        (ps, true)
    } else {
        let prompt = read_prompt(args)?;
        (vec![prompt], false)
    };

    // Compute effective context length for KV cache auto-sizing.
    let model_max = model.max_sequence_length();
    let effective_ctx = if args.interactive {
        // Interactive conversations need all available context.
        match args.context_len {
            Some(cl) => cl.min(model_max),
            None => model_max,
        }
    } else {
        match args.context_len {
            Some(cl) => cl.min(model_max),
            None => {
                let max_prompt_tokens = prompts
                    .iter()
                    .map(|p| tokenizer.encode(p).map(|t| t.len()).unwrap_or(0))
                    .max()
                    .unwrap_or(0);
                // Margin: extra 128 if chat template may add tokens, else 64
                let margin = if use_chat { 128 } else { 64 };
                (max_prompt_tokens + args.max_tokens + margin).min(model_max)
            }
        }
    };

    // Print effective KV cache size.
    let n_kv_heads = model.num_kv_heads();
    let head_dim = model.head_dim();
    let n_layers = model.num_layers();
    let cache_bytes = 2 * n_layers * n_kv_heads * effective_ctx * head_dim * 4;
    let cache_mb = cache_bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "  KV cache: {:.1} MB ({}layers x {}heads x {}seq x {}dim x f32 x 2)",
        cache_mb, n_layers, n_kv_heads, effective_ctx, head_dim
    );

    let mut generator = Generator::new(model, tokenizer, args.temperature);
    generator = generator.with_context_len(effective_ctx);
    generator = generator.with_optimization_profile(profile);

    if let Some(k) = args.top_k {
        generator = generator.with_top_k(k);
    }
    if let Some(p) = args.top_p {
        generator = generator.with_top_p(p);
    }
    if let Some(rp) = args.repetition_penalty {
        generator = generator.with_repetition_penalty(rp);
    }
    if let Some(eos) = eos_token_id {
        generator = generator.with_eos_token(eos);
    }
    if let Some(bos) = bos_token_id {
        generator = generator.with_bos_token(bos);
    }
    if use_chat {
        generator = generator.with_chat_template(chat_template);
    }
    if let Some(secs) = args.timeout {
        generator = generator.with_timeout(Duration::from_secs_f64(secs));
        eprintln!("  Timeout: {:.1}s", secs);
    }

    if args.interactive {
        return run_interactive(&generator, tokenizer, args.max_tokens);
    }

    if is_batch {
        let prompt_refs: Vec<&str> = prompts.iter().map(|s| s.as_str()).collect();
        eprintln!("  Batch: {} prompts", prompt_refs.len());
        eprintln!("---");

        let gen_start = Instant::now();
        let results = generator.generate_batch_parallel(&prompt_refs, args.max_tokens)?;
        let elapsed = gen_start.elapsed();

        for (i, output) in results.iter().enumerate() {
            println!("[{}] {}", i, output);
        }
        eprintln!("---");
        eprintln!(
            "  {} prompts in {:.2}s ({:.1} prompts/sec)",
            results.len(),
            elapsed.as_secs_f64(),
            results.len() as f64 / elapsed.as_secs_f64().max(1e-9)
        );
    } else {
        let prompt = &prompts[0];
        eprintln!("---");

        let gen_start = Instant::now();

        if args.stream {
            let mut token_count: usize = 0;
            let _output =
                generator.generate_stream(prompt, args.max_tokens, |token_id| {
                    token_count += 1;
                    match tokenizer.decode(&[token_id]) {
                        Ok(piece) => print!("{piece}"),
                        Err(e) => {
                            eprintln!("[warn] failed to decode token {}: {}", token_id, e)
                        }
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
            eprintln!(
                "  {} tokens in {:.2}s ({:.1} tokens/sec)",
                token_count,
                elapsed.as_secs_f64(),
                tps
            );
        } else {
            let output = generator.generate(prompt, args.max_tokens)?;
            let elapsed = gen_start.elapsed();
            println!("{output}");
            eprintln!("---");
            eprintln!("  Generated in {:.2}s", elapsed.as_secs_f64());
        }
    }

    Ok(())
}

fn run_interactive(
    generator: &rustml_nlp::Generator,
    tokenizer: &(dyn rustml_tokenizer::Tokenizer + Sync),
    max_tokens: usize,
) -> Result<()> {
    eprintln!("Interactive chat mode. Type 'quit' or 'exit' to leave, '/clear' to reset history.");
    eprintln!("---");

    let mut history: Vec<(String, String)> = Vec::new();
    let stdin = io::stdin();

    loop {
        eprint!("> ");
        io::stderr().flush().ok();

        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => {
                // EOF (Ctrl-D)
                eprintln!("\nGoodbye!");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "quit" | "exit" => {
                eprintln!("Goodbye!");
                break;
            }
            "/clear" => {
                history.clear();
                eprintln!("[History cleared]");
                continue;
            }
            _ => {}
        }

        // Build messages from history + current input
        let mut messages: Vec<(&str, &str)> = Vec::new();
        for (role, content) in &history {
            messages.push((role.as_str(), content.as_str()));
        }
        messages.push(("user", input));

        let gen_start = Instant::now();
        let mut token_count: usize = 0;

        let response = generator.generate_turn_stream(&messages, max_tokens, |token_id| {
            token_count += 1;
            match tokenizer.decode(&[token_id]) {
                Ok(piece) => {
                    print!("{piece}");
                    io::stdout().flush().ok();
                }
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
        eprintln!(
            "  [{} tokens in {:.2}s ({:.1} tok/s)]",
            token_count,
            elapsed.as_secs_f64(),
            tps,
        );

        // Append to history
        history.push(("user".to_string(), input.to_string()));
        history.push(("assistant".to_string(), response));
    }

    Ok(())
}

pub fn run(args: InferArgs) -> Result<()> {
    let batch_contents = validate_args(&args)?;
    let profile = parse_opt_profile(&args.opt_profile)?;

    // Apply runtime config (rayon thresholds) for this profile
    profile.runtime_config().apply()
        .map_err(|e| anyhow::anyhow!("Failed to apply runtime config: {}", e))?;
    eprintln!("  Optimization profile: {:?}", profile);

    if let Some(ref model_id) = args.safetensors {
        run_safetensors(model_id, &args, batch_contents, profile)
    } else {
        let gguf_path = args.gguf_path.as_ref().unwrap();
        run_gguf(gguf_path, &args, batch_contents, profile)
    }
}

fn run_safetensors(
    model_id: &str,
    args: &InferArgs,
    batch_contents: Option<String>,
    profile: OptProfile,
) -> Result<()> {
    let hub = HubApi::new();
    let bundle = match hub.get_cached(model_id) {
        Some(b) => {
            eprintln!("Using cached model: {}", model_id);
            b
        }
        None => {
            eprintln!("Downloading model: {}", model_id);
            hub.download_model_sync(model_id)
                .with_context(|| format!("Failed to download model: {}", model_id))?
        }
    };

    let json_config = bundle
        .load_config_sync()
        .with_context(|| "Failed to load config.json")?;
    let model_type = json_config["model_type"].as_str().unwrap_or("").to_string();
    let config = rustml_nlp::ModelConfig::from_json_value(&json_config)
        .with_context(|| "Failed to parse model config")?;
    eprintln!(
        "  Config: arch={}, dim={}, layers={}, heads={}, vocab={}",
        if model_type.is_empty() { "gpt2" } else { &model_type },
        config.dim, config.n_layers, config.n_heads, config.vocab_size
    );

    eprintln!("  Loading SafeTensors weights...");
    let weights = bundle
        .load_tensors()
        .with_context(|| "Failed to load SafeTensors weights")?;
    eprintln!("  {} tensors loaded", weights.len());

    eprintln!("  Building model...");
    let mut model = rustml_nlp::build_safetensors_model(&model_type, &config, weights)
        .with_context(|| format!("Failed to build {} model", if model_type.is_empty() { "gpt2" } else { &model_type }))?;
    model.set_optimization_profile(profile);

    // Quantize all F32 linear layers to Q8_0 for reduced memory bandwidth
    if !model.output.is_quantized() {
        match model.quantize_all_weights(None) {
            Ok(n) if n > 0 => eprintln!("  Quantized {} linear layers F32 -> Q8_0", n),
            Ok(_) => {}
            Err(e) => eprintln!("  [warn] weight quantization failed: {}", e),
        }
    }

    // Fuse gate+up projections for gated activations (SwiGLU, GeGLU)
    let fused = model.fuse_gate_up_weights();
    if fused > 0 {
        eprintln!("  Fused {} gate+up projection pairs", fused);
    }

    // Fuse Q+K+V projections in attention layers
    let fused_qkv = model.fuse_qkv_weights();
    if fused_qkv > 0 {
        eprintln!("  Fused {} QKV projection triples", fused_qkv);
    }

    // Warm up M=1 decode path (rayon, SIMD, branch prediction, TLB)
    let warmup_start = Instant::now();
    if let Err(e) = model.warmup_decode() {
        eprintln!("  [warn] decode warmup failed: {}", e);
    }
    eprintln!("  Warmup: {:.0}ms", warmup_start.elapsed().as_secs_f64() * 1000.0);

    let (total_params, _) = model.parameter_count();
    eprintln!("  Model ready: {:.1}M params", total_params as f64 / 1e6);

    // Use HFTokenizer if tokenizer.json exists, otherwise fall back to BPE (GPT-2)
    let tokenizer_json = bundle.tokenizer_json_path();
    let tokenizer: Box<dyn Tokenizer + Sync> = if tokenizer_json.exists() {
        let hf = HFTokenizer::from_file(&tokenizer_json)
            .with_context(|| "Failed to load HFTokenizer from tokenizer.json")?;
        eprintln!("  Tokenizer: {} tokens (tokenizer.json)", hf.vocab_size());
        Box::new(hf)
    } else {
        let bpe = BpeTokenizer::from_files(bundle.vocab_path(), bundle.merges_path())
            .with_context(|| "Failed to load BPE tokenizer")?;
        eprintln!("  Tokenizer: {} tokens (BPE)", bpe.vocab_size());
        Box::new(bpe)
    };

    let eos = config.eos_token_id.or_else(|| {
        // Fall back to GPT-2 EOS if no eos_token_id in config (legacy GPT-2 models)
        if model_type.is_empty() || model_type == "gpt2" {
            Some(BpeTokenizer::GPT2_EOS_TOKEN_ID)
        } else {
            None
        }
    });

    run_generation(
        &model,
        tokenizer.as_ref(),
        args,
        batch_contents,
        eos,
        config.bos_token_id,
        config.chat_template.clone(),
        profile,
    )
}

fn run_gguf(
    gguf_path: &PathBuf,
    args: &InferArgs,
    batch_contents: Option<String>,
    profile: OptProfile,
) -> Result<()> {
    eprintln!("Loading GGUF: {}", gguf_path.display());
    let gguf = GGUFFile::parse_header(gguf_path)
        .with_context(|| format!("Failed to parse GGUF: {}", gguf_path.display()))?;
    eprintln!(
        "  GGUF v{}, {} tensors",
        gguf.version,
        gguf.tensor_infos.len()
    );

    let gguf_config = gguf
        .to_model_config()
        .with_context(|| "Failed to extract model config from GGUF")?;
    let config = gguf_config_to_model_config(&gguf_config)
        .with_context(|| "Failed to convert GGUF config to model config")?;
    eprintln!(
        "  arch={}, dim={}, layers={}, heads={}, vocab={}",
        gguf_config.architecture, config.dim, config.n_layers, config.n_heads, config.vocab_size
    );

    let tokenizer = GgufTokenizer::from_gguf(&gguf)
        .with_context(|| "Failed to build tokenizer from GGUF")?;
    eprintln!("  Tokenizer: {} tokens", tokenizer.vocab_size());

    eprintln!("  Loading tensors...");
    let is_gemma3 = gguf_config.architecture == "gemma3";
    let loaded_tensors = if is_gemma3 {
        gguf.load_and_remap_gemma3(gguf_path, config.n_layers)
            .with_context(|| "Failed to load/remap gemma3 tensors")?
    } else {
        gguf.load_and_remap(gguf_path, config.n_layers)
            .with_context(|| "Failed to load/remap tensors")?
    };
    let tensors = convert_tensors(loaded_tensors);
    eprintln!("  {} tensors loaded", tensors.len());

    eprintln!("  Building model...");
    let mut model = if is_gemma3 {
        LlmModel::from_pretrained_gemma3(&config, tensors)
            .with_context(|| "Failed to build gemma3 model")?
    } else {
        LlmModel::from_pretrained(&config, tensors)
            .with_context(|| "Failed to build model")?
    };
    model.set_optimization_profile(profile);

    // Fuse Q+K+V projections in attention layers
    let fused_qkv = model.fuse_qkv_weights();
    if fused_qkv > 0 {
        eprintln!("  Fused {} QKV projection triples", fused_qkv);
    }

    // Warm up M=1 decode path (rayon, SIMD, branch prediction, TLB)
    let warmup_start = Instant::now();
    if let Err(e) = model.warmup_decode() {
        eprintln!("  [warn] decode warmup failed: {}", e);
    }
    eprintln!("  Warmup: {:.0}ms", warmup_start.elapsed().as_secs_f64() * 1000.0);

    let (total_params, _) = model.parameter_count();
    eprintln!("  Model ready: {:.1}M params", total_params as f64 / 1e6);

    run_generation(
        &model,
        &tokenizer,
        args,
        batch_contents,
        config.eos_token_id,
        config.bos_token_id,
        config.chat_template.clone(),
        profile,
    )
}
