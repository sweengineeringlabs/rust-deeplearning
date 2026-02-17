use anyhow::{Context, Result};

use crate::args::DownloadArgs;

const DEFAULT_FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];

pub fn execute(args: &DownloadArgs) -> Result<()> {
    let api = hf_hub::api::sync::Api::new().context("Failed to initialize HuggingFace API")?;
    let repo = api.model(args.repo.clone());

    let files: Vec<&str> = if let Some(ref f) = args.files {
        f.iter().map(|s| s.as_str()).collect()
    } else {
        DEFAULT_FILES.to_vec()
    };

    eprintln!("Downloading from {} ...", args.repo);

    for filename in &files {
        eprint!("  {} ... ", filename);
        match repo.get(filename) {
            Ok(path) => {
                eprintln!("{}", path.display());
            }
            Err(e) => {
                eprintln!("FAILED: {}", e);
            }
        }
    }

    eprintln!("Done.");
    Ok(())
}
