use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use rustml_gguf::{GGUFFile, GGUFValue};

/// RustML GGUF Inspector — inspect GGUF model files.
#[derive(Parser)]
#[command(name = "rustml-gguf-inspect", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show model summary: version, architecture, tensor count, dimensions, vocab size.
    Info {
        /// Path to the GGUF model file.
        path: PathBuf,
    },

    /// Dump metadata key-value pairs.
    Meta {
        /// Path to the GGUF model file.
        path: PathBuf,

        /// Filter to a specific metadata key.
        #[arg(long)]
        key: Option<String>,
    },

    /// List tensor information: name, dtype, shape, offset.
    Tensors {
        /// Path to the GGUF model file.
        path: PathBuf,

        /// Filter tensors whose name contains this pattern.
        #[arg(long)]
        filter: Option<String>,
    },
}

/// Format a GGUFValue for display: scalars inline, strings truncated at 200 chars,
/// arrays as `[array, len=N]`.
fn format_gguf_value(val: &GGUFValue) -> String {
    match val {
        GGUFValue::U8(v) => format!("{v}"),
        GGUFValue::I8(v) => format!("{v}"),
        GGUFValue::U16(v) => format!("{v}"),
        GGUFValue::I16(v) => format!("{v}"),
        GGUFValue::U32(v) => format!("{v}"),
        GGUFValue::I32(v) => format!("{v}"),
        GGUFValue::U64(v) => format!("{v}"),
        GGUFValue::I64(v) => format!("{v}"),
        GGUFValue::F32(v) => format!("{v}"),
        GGUFValue::F64(v) => format!("{v}"),
        GGUFValue::Bool(v) => format!("{v}"),
        GGUFValue::String(v) => {
            if v.len() > 200 {
                format!("\"{}...\" (len={})", &v[..200], v.len())
            } else {
                format!("\"{v}\"")
            }
        }
        GGUFValue::Array(arr) => format!("[array, len={}]", arr.len()),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Command::Info { path } => {
            let gguf = GGUFFile::parse_header(path)
                .with_context(|| format!("Failed to parse GGUF: {}", path.display()))?;

            println!("GGUF version:  {}", gguf.version);
            println!("Tensor count:  {}", gguf.tensor_infos.len());
            println!("Data offset:   {}", gguf.data_offset);

            match gguf.to_model_config() {
                Ok(config) => {
                    println!("Architecture:  {}", config.architecture);
                    println!("Dimensions:    {}", config.dim);
                    println!("Hidden dim:    {}", config.hidden_dim);
                    println!("Layers:        {}", config.n_layers);
                    println!("Heads:         {}", config.n_heads);
                    if let Some(kv) = config.n_kv_heads {
                        println!("KV heads:      {kv}");
                    }
                    println!("Vocab size:    {}", config.vocab_size);
                    println!("Max seq len:   {}", config.max_seq_len);
                    println!("Norm eps:      {}", config.norm_eps);
                    println!("RoPE theta:    {}", config.rope_theta);
                    if let Some(hd) = config.head_dim {
                        println!("Head dim:      {hd}");
                    }
                    if let Some(sw) = config.sliding_window {
                        println!("Sliding win:   {sw}");
                    }
                }
                Err(e) => {
                    eprintln!("Could not extract model config: {e}");
                }
            }
        }

        Command::Meta { path, key } => {
            let gguf = GGUFFile::parse_header(path)
                .with_context(|| format!("Failed to parse GGUF: {}", path.display()))?;

            if let Some(filter_key) = key {
                match gguf.metadata.get(filter_key) {
                    Some(val) => println!("{filter_key} = {}", format_gguf_value(val)),
                    None => eprintln!("Key not found: {filter_key}"),
                }
            } else {
                let mut keys: Vec<&String> = gguf.metadata.keys().collect();
                keys.sort();
                for k in keys {
                    println!("{k} = {}", format_gguf_value(&gguf.metadata[k]));
                }
            }
        }

        Command::Tensors { path, filter } => {
            let gguf = GGUFFile::parse_header(path)
                .with_context(|| format!("Failed to parse GGUF: {}", path.display()))?;

            let iter = gguf.tensor_infos.iter().filter(|t| {
                filter
                    .as_ref()
                    .map_or(true, |pat| t.name.contains(pat.as_str()))
            });

            let mut count = 0usize;
            for info in iter {
                println!(
                    "{:<60} {:?}  {:?}  offset={}",
                    info.name, info.ggml_type, info.dimensions, info.offset
                );
                count += 1;
            }
            eprintln!("{count} tensor(s) listed");
        }
    }

    Ok(())
}
