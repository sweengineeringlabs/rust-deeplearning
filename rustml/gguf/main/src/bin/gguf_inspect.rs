use std::path::PathBuf;
use std::process;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use rustml_gguf::{GGUFFile, GGUFValue, TensorStats};

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

        /// Compute and display min/max/mean/std for each tensor.
        #[arg(long)]
        stats: bool,

        /// Print first N dequantized f32 values per tensor.
        #[arg(long)]
        head: Option<usize>,
    },

    /// Verify tokenizer-weight contract checks (header-only).
    Verify {
        /// Path to the GGUF model file.
        path: PathBuf,
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

/// Canonical per-layer tensor suffixes for llama-family models.
const LAYER_SUFFIXES: &[&str] = &[
    "attn_norm.weight",
    "ffn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

fn run_verify(path: &PathBuf) -> Result<()> {
    let gguf = GGUFFile::parse_header(path)
        .with_context(|| format!("Failed to parse GGUF: {}", path.display()))?;

    let config = gguf.to_model_config()
        .with_context(|| "Cannot extract model config (missing architecture metadata)")?;

    let tensor_names: std::collections::HashSet<&str> = gguf
        .tensor_infos
        .iter()
        .map(|t| t.name.as_str())
        .collect();

    let find_tensor = |name: &str| -> Option<&rustml_gguf::GGUFTensorInfo> {
        gguf.tensor_infos.iter().find(|t| t.name == name)
    };

    let mut fails = 0u32;
    let mut warns = 0u32;
    let mut passes = 0u32;

    // Check 1: token_embd.weight exists
    let embd = find_tensor("token_embd.weight");
    if embd.is_some() {
        println!("PASS  token_embd.weight exists");
        passes += 1;
    } else {
        println!("FAIL  token_embd.weight missing");
        fails += 1;
    }

    // Check 2: token_embd.weight shape = [dim, vocab_size] (GGUF inner-first)
    if let Some(info) = embd {
        if info.dimensions.len() == 2 && info.dimensions[0] == config.dim {
            println!("PASS  token_embd.weight shape [{}x{}] matches dim={}", info.dimensions[0], info.dimensions[1], config.dim);
            passes += 1;
        } else {
            println!("FAIL  token_embd.weight shape {:?} — expected dim[0]={}", info.dimensions, config.dim);
            fails += 1;
        }
    }

    // Check 3: vocab_size matches token_embd.weight row count
    if let Some(info) = embd {
        if info.dimensions.len() == 2 {
            let embd_vocab = info.dimensions[1];
            if embd_vocab == config.vocab_size {
                println!("PASS  vocab_size={} matches token_embd.weight[1]", config.vocab_size);
                passes += 1;
            } else {
                println!("FAIL  vocab_size={} != token_embd.weight[1]={}", config.vocab_size, embd_vocab);
                fails += 1;
            }
        }
    }

    // Check 4: output.weight exists (WARN if absent — tied embeddings)
    let output = find_tensor("output.weight");
    if output.is_some() {
        println!("PASS  output.weight exists");
        passes += 1;
    } else {
        println!("WARN  output.weight missing (tied embeddings?)");
        warns += 1;
    }

    // Check 5: output.weight shape matches [dim, vocab_size]
    if let Some(info) = output {
        if info.dimensions.len() == 2
            && info.dimensions[0] == config.dim
            && info.dimensions[1] == config.vocab_size
        {
            println!("PASS  output.weight shape [{}x{}]", info.dimensions[0], info.dimensions[1]);
            passes += 1;
        } else {
            println!("FAIL  output.weight shape {:?} — expected [{}x{}]", info.dimensions, config.dim, config.vocab_size);
            fails += 1;
        }
    }

    // Check 6: All per-layer tensors for blk.0..blk.{n_layers-1}
    let mut layer_ok = true;
    for layer in 0..config.n_layers {
        for suffix in LAYER_SUFFIXES {
            let name = format!("blk.{layer}.{suffix}");
            if !tensor_names.contains(name.as_str()) {
                println!("FAIL  missing {name}");
                fails += 1;
                layer_ok = false;
            }
        }
    }
    if layer_ok {
        println!("PASS  all layer tensors present for {} layers ({} suffixes each)", config.n_layers, LAYER_SUFFIXES.len());
        passes += 1;
    }

    // Check 7: Embedding dim consistency across attention projections
    let mut dim_ok = true;
    for layer in 0..config.n_layers {
        for suffix in &["attn_q.weight", "attn_output.weight"] {
            let name = format!("blk.{layer}.{suffix}");
            if let Some(info) = find_tensor(&name) {
                if info.dimensions.is_empty() || info.dimensions[0] != config.dim {
                    println!("FAIL  {name} dim[0]={} != embedding_dim={}", info.dimensions.first().unwrap_or(&0), config.dim);
                    fails += 1;
                    dim_ok = false;
                }
            }
        }
    }
    if dim_ok {
        println!("PASS  embedding dim consistent across attention projections");
        passes += 1;
    }

    // Summary
    println!("\n--- Summary: {passes} PASS, {fails} FAIL, {warns} WARN ---");

    if fails > 0 {
        process::exit(1);
    }

    Ok(())
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

        Command::Tensors { path, filter, stats, head } => {
            let gguf = GGUFFile::parse_header(path)
                .with_context(|| format!("Failed to parse GGUF: {}", path.display()))?;

            let need_data = *stats || head.is_some();

            let filtered: Vec<_> = gguf.tensor_infos.iter().filter(|t| {
                filter
                    .as_ref()
                    .map_or(true, |pat| t.name.contains(pat.as_str()))
            }).collect();

            let mut count = 0usize;
            for info in &filtered {
                println!(
                    "{:<60} {:?}  {:?}  offset={}",
                    info.name, info.ggml_type, info.dimensions, info.offset
                );

                if need_data {
                    match GGUFFile::read_tensor_f32(path, info, gguf.data_offset) {
                        Ok(values) => {
                            if *stats {
                                if let Some(s) = TensorStats::compute(&values) {
                                    println!(
                                        "  stats: min={:.6} max={:.6} mean={:.6} std={:.6} n={}",
                                        s.min, s.max, s.mean, s.std, s.n_elements
                                    );
                                }
                            }
                            if let Some(n) = head {
                                let n = (*n).min(values.len());
                                let slice = &values[..n];
                                println!("  head({n}): {:?}", slice);
                            }
                        }
                        Err(e) => {
                            eprintln!("  error reading tensor data: {e}");
                        }
                    }
                }

                count += 1;
            }
            eprintln!("{count} tensor(s) listed");
        }

        Command::Verify { path } => {
            run_verify(path)?;
        }
    }

    Ok(())
}
