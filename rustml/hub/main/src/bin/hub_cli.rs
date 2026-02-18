use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use rustml_hub::HubApi;

/// RustML Hub CLI — download, cache, and inspect HuggingFace models.
#[derive(Parser)]
#[command(name = "rustml-hub-cli", version, about)]
struct Cli {
    /// Override the default cache directory.
    #[arg(long, global = true)]
    cache_dir: Option<PathBuf>,

    /// HuggingFace API token for private models.
    #[arg(long, global = true)]
    token: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download a model from HuggingFace Hub.
    Download {
        /// Model identifier (e.g. "openai-community/gpt2").
        model_id: String,

        /// Download a GGUF file instead of SafeTensors.
        /// Provide the filename within the repo (e.g. "model-Q4_0.gguf").
        #[arg(long)]
        gguf: Option<String>,
    },

    /// List cached models in the local cache directory.
    List,

    /// Show config.json for a cached model.
    Info {
        /// Model identifier (e.g. "openai-community/gpt2").
        model_id: String,
    },
}

fn build_api(cli: &Cli) -> HubApi {
    let mut api = match &cli.cache_dir {
        Some(dir) => HubApi::with_cache_dir(dir),
        None => HubApi::new(),
    };
    if let Some(ref token) = cli.token {
        api = api.with_token(token);
    }
    api
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let api = build_api(&cli);

    match &cli.command {
        Command::Download { model_id, gguf } => {
            if let Some(filename) = gguf {
                eprintln!("Downloading GGUF {model_id} / {filename} ...");
                let bundle = api
                    .download_gguf_sync(model_id, filename)
                    .with_context(|| format!("Failed to download GGUF: {model_id}/{filename}"))?;
                println!("{}", bundle.gguf_path.display());
            } else {
                eprintln!("Downloading model {model_id} ...");
                let bundle = api
                    .download_model_sync(model_id)
                    .with_context(|| format!("Failed to download model: {model_id}"))?;
                println!("{}", bundle.model_dir.display());
            }
        }

        Command::List => {
            let cache = api.cache_dir();
            if !cache.exists() {
                eprintln!("Cache directory does not exist: {}", cache.display());
                return Ok(());
            }
            let mut found = false;
            let entries = std::fs::read_dir(cache)
                .with_context(|| format!("Failed to read cache dir: {}", cache.display()))?;
            for entry in entries {
                let entry = entry?;
                // Use path().is_dir() to follow symlinks (e.g. cache entries
                // created by download_model_sync linking to hf-hub snapshots).
                if entry.path().is_dir() {
                    let dir_name = entry.file_name();
                    let dir_str = dir_name.to_string_lossy();
                    // Reconstruct model ID: directory names use "--" for "/"
                    let model_id = dir_str.replacen("--", "/", 1);
                    println!("{model_id}");
                    found = true;
                }
            }
            if !found {
                eprintln!("No cached models found.");
            }
        }

        Command::Info { model_id } => {
            let bundle = api
                .get_cached(model_id)
                .with_context(|| format!("Model not cached: {model_id}"))?;
            let config = bundle
                .load_config_sync()
                .with_context(|| format!("Failed to load config for {model_id}"))?;
            println!("{}", serde_json::to_string_pretty(&config)?);
        }
    }

    Ok(())
}
