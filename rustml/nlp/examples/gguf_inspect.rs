//! Inspect GGUF file metadata and tensor info.
//! Usage: cargo run -p rustml-nlp --example gguf_inspect -- /path/to/model.gguf

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/tmp/gemma3-gguf/gemma-3-1b-it-Q4_0.gguf".to_string()
    });

    println!("Parsing GGUF: {}", path);
    let gguf = rustml_gguf::GGUFFile::parse_header(&path).unwrap();
    println!("Version: {}", gguf.version);
    println!("Tensors: {}", gguf.tensor_infos.len());

    println!("\n=== Metadata Keys ===");
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for key in &keys {
        let val = &gguf.metadata[*key];
        let val_str = match val {
            rustml_gguf::GGUFValue::U32(v) => format!("{}", v),
            rustml_gguf::GGUFValue::I32(v) => format!("{}", v),
            rustml_gguf::GGUFValue::F32(v) => format!("{}", v),
            rustml_gguf::GGUFValue::F64(v) => format!("{}", v),
            rustml_gguf::GGUFValue::U64(v) => format!("{}", v),
            rustml_gguf::GGUFValue::Bool(v) => format!("{}", v),
            rustml_gguf::GGUFValue::String(v) => {
                if v.len() > 200 {
                    format!("\"{}...\" (len={})", &v[..200], v.len())
                } else {
                    format!("\"{}\"", v)
                }
            }
            rustml_gguf::GGUFValue::Array(arr) => format!("[array, len={}]", arr.len()),
            _ => format!("{:?}", val),
        };
        println!("  {} = {}", key, val_str);
    }

    println!("\n=== Tensor Info (first 40) ===");
    for (i, info) in gguf.tensor_infos.iter().enumerate() {
        if i >= 40 {
            println!("  ... and {} more", gguf.tensor_infos.len() - 40);
            break;
        }
        println!(
            "  {} [{:?}] {:?}",
            info.name, info.ggml_type, info.dimensions
        );
    }

    println!("\n=== Parsed Model Config ===");
    match gguf.to_model_config() {
        Ok(config) => println!("{:#?}", config),
        Err(e) => println!("Config error: {:?}", e),
    }
}
