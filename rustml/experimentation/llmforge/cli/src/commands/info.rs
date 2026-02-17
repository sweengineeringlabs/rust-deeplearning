use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

use llmforge::loader::gguf::GGUFFile;
use llmforge::loader::ModelLoader;

use crate::args::InfoArgs;
use crate::model_source::{self, ModelFormat};

pub fn execute(args: &InfoArgs) -> Result<()> {
    let resolved = model_source::resolve(
        &args.model,
        args.file.as_deref(),
        None,
        None,
    )?;

    eprintln!("Inspecting: {}", resolved.model_path.display());

    match resolved.format {
        ModelFormat::Gguf => inspect_gguf(&resolved.model_path)?,
        ModelFormat::SafeTensors => inspect_safetensors(&resolved.model_path)?,
    }

    Ok(())
}

fn inspect_gguf(path: &Path) -> Result<()> {
    let gguf = GGUFFile::parse_header(path).context("Failed to parse GGUF header")?;

    println!("Format: GGUF v{}", gguf.version);
    println!("Tensors: {}", gguf.tensor_infos.len());
    println!();

    // Print metadata
    println!("--- Metadata ---");
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for key in keys {
        let value = &gguf.metadata[key];
        println!("  {}: {}", key, format_gguf_value(value));
    }

    // Print model config if extractable
    println!();
    match gguf.to_model_config() {
        Ok(config) => {
            println!("--- Model Config ---");
            println!("  Dimension    : {}", config.dim);
            println!("  Hidden dim   : {}", config.hidden_dim);
            println!("  Layers       : {}", config.n_layers);
            println!("  Heads        : {}", config.n_heads);
            println!("  KV Heads     : {}", config.n_kv_heads.unwrap_or(config.n_heads));
            println!("  Vocab size   : {}", config.vocab_size);
            println!("  Max seq len  : {}", config.max_seq_len);
            println!("  Norm epsilon : {}", config.norm_eps);
            println!("  RoPE theta   : {}", config.rope_theta);
            println!("  Position enc : {:?}", config.position_encoding);
        }
        Err(e) => {
            eprintln!("  (Could not extract model config: {})", e);
        }
    }

    // Print tensor list with quant types
    println!();
    println!("--- Tensors ---");
    let mut quant_counts: HashMap<String, usize> = HashMap::new();
    for info in &gguf.tensor_infos {
        let dims: Vec<String> = info.dimensions.iter().map(|d| d.to_string()).collect();
        let type_name = format!("{:?}", info.ggml_type);
        println!("  {:50} {:?}  [{}]", info.name, info.ggml_type, dims.join(", "));
        *quant_counts.entry(type_name).or_insert(0) += 1;
    }

    println!();
    println!("--- Quantization Breakdown ---");
    let mut counts: Vec<(String, usize)> = quant_counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    for (qtype, count) in &counts {
        println!("  {:8}: {} tensors", qtype, count);
    }

    Ok(())
}

fn inspect_safetensors(path: &Path) -> Result<()> {
    let weights = ModelLoader::load_safetensors(path)
        .context("Failed to load SafeTensors file")?;

    println!("Format: SafeTensors");
    println!("Tensors: {}", weights.len());
    println!();

    println!("--- Tensors ---");
    let mut names: Vec<&String> = weights.keys().collect();
    names.sort();
    for name in &names {
        let tensor = &weights[*name];
        let shape: Vec<String> = tensor.shape().iter().map(|d| d.to_string()).collect();
        println!(
            "  {:50} {:?}  [{}]  ({} elements)",
            name,
            tensor.dtype(),
            shape.join(", "),
            tensor.element_count()
        );
    }

    // Dtype breakdown
    println!();
    println!("--- DType Breakdown ---");
    let mut dtype_counts: HashMap<String, usize> = HashMap::new();
    for tensor in weights.values() {
        let name = format!("{:?}", tensor.dtype());
        *dtype_counts.entry(name).or_insert(0) += 1;
    }
    let mut counts: Vec<(String, usize)> = dtype_counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    for (dtype, count) in &counts {
        println!("  {:8}: {} tensors", dtype, count);
    }

    // Total parameter count
    let total_elements: usize = weights.values().map(|t| t.element_count()).sum();
    println!();
    println!("Total parameters: {:.1}M", total_elements as f64 / 1e6);

    Ok(())
}

fn format_gguf_value(value: &llmforge::loader::gguf::GGUFValue) -> String {
    use llmforge::loader::gguf::GGUFValue;
    match value {
        GGUFValue::U8(v) => format!("{}", v),
        GGUFValue::I8(v) => format!("{}", v),
        GGUFValue::U16(v) => format!("{}", v),
        GGUFValue::I16(v) => format!("{}", v),
        GGUFValue::U32(v) => format!("{}", v),
        GGUFValue::I32(v) => format!("{}", v),
        GGUFValue::U64(v) => format!("{}", v),
        GGUFValue::I64(v) => format!("{}", v),
        GGUFValue::F32(v) => format!("{}", v),
        GGUFValue::F64(v) => format!("{}", v),
        GGUFValue::Bool(v) => format!("{}", v),
        GGUFValue::String(v) => format!("\"{}\"", v),
        GGUFValue::Array(arr) => format!("[{} elements]", arr.len()),
    }
}
