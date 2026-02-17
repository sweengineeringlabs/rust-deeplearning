use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "llmforge", about = "LLMForge — local LLM inference engine")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Load a model and generate text
    Run(RunArgs),
    /// Inspect a model file (architecture, quant stats, tensor list)
    Info(InfoArgs),
    /// Pre-download a HuggingFace model repo
    Download(DownloadArgs),
}

#[derive(Clone, ValueEnum)]
pub enum Arch {
    Auto,
    Gpt2,
    Llama,
}

#[derive(Parser)]
pub struct RunArgs {
    /// Local path (.gguf/.safetensors), directory, or HuggingFace repo ID
    #[arg(long)]
    pub model: String,

    /// Specific file from HF repo (required for GGUF repos)
    #[arg(long)]
    pub file: Option<String>,

    /// Path to tokenizer.json (auto-detected if omitted)
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// Path to config.json (SafeTensors only, auto-detected if omitted)
    #[arg(long)]
    pub config: Option<String>,

    /// Architecture hint for SafeTensors weight mapping
    #[arg(long, value_enum, default_value = "auto")]
    pub arch: Arch,

    /// Single-shot prompt (omit for interactive REPL)
    #[arg(long)]
    pub prompt: Option<String>,

    /// Force interactive REPL mode
    #[arg(long)]
    pub interactive: bool,

    /// Sampling temperature
    #[arg(long, default_value = "0.8")]
    pub temperature: f32,

    /// Top-k sampling
    #[arg(long, default_value = "40")]
    pub top_k: usize,

    /// Nucleus sampling threshold
    #[arg(long, default_value = "0.95")]
    pub top_p: f32,

    /// Repetition penalty (1.0 = disabled)
    #[arg(long, default_value = "1.0")]
    pub repetition_penalty: f32,

    /// Maximum generation length in tokens
    #[arg(long, default_value = "128")]
    pub max_tokens: usize,

    /// CPU thread count (0 = auto-detect)
    #[arg(long, default_value = "0")]
    pub threads: usize,
}

#[derive(Parser)]
pub struct InfoArgs {
    /// Local path or HuggingFace repo ID
    #[arg(long)]
    pub model: String,

    /// Specific file from HF repo
    #[arg(long)]
    pub file: Option<String>,
}

#[derive(Parser)]
pub struct DownloadArgs {
    /// HuggingFace repo ID (e.g. openai-community/gpt2)
    pub repo: String,

    /// Specific files to download (default: config.json, tokenizer.json, model.safetensors)
    #[arg(long, num_args = 1..)]
    pub files: Option<Vec<String>>,
}
