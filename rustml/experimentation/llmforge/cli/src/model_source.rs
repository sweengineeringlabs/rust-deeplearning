use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// Detected model file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
}

/// A fully resolved model source with local paths.
#[derive(Debug)]
pub struct ResolvedModel {
    pub model_path: PathBuf,
    pub format: ModelFormat,
    pub config_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
}

/// Resolve `--model` (and optional `--file`) to local paths.
///
/// Three cases:
///  1. Local file  — detect format by extension, look for sibling config/tokenizer
///  2. Local directory — scan for .gguf or model.safetensors inside
///  3. HuggingFace repo ID — pattern `org/name` that doesn't exist on disk
pub fn resolve(
    model: &str,
    file: Option<&str>,
    config_override: Option<&str>,
    tokenizer_override: Option<&str>,
) -> Result<ResolvedModel> {
    let path = Path::new(model);

    if path.is_file() {
        resolve_local_file(path, config_override, tokenizer_override)
    } else if path.is_dir() {
        resolve_local_dir(path, config_override, tokenizer_override)
    } else if looks_like_hf_repo(model) {
        resolve_hf_repo(model, file, config_override, tokenizer_override)
    } else {
        bail!("'{}' is not an existing file/directory and doesn't look like a HuggingFace repo ID (expected org/name)", model)
    }
}

fn resolve_local_file(
    path: &Path,
    config_override: Option<&str>,
    tokenizer_override: Option<&str>,
) -> Result<ResolvedModel> {
    let format = detect_format(path)?;
    let parent = path.parent().unwrap_or(Path::new("."));

    let config_path = config_override
        .map(PathBuf::from)
        .or_else(|| {
            let p = parent.join("config.json");
            p.is_file().then_some(p)
        });

    let tokenizer_path = tokenizer_override
        .map(PathBuf::from)
        .or_else(|| {
            let p = parent.join("tokenizer.json");
            p.is_file().then_some(p)
        });

    Ok(ResolvedModel {
        model_path: path.to_path_buf(),
        format,
        config_path,
        tokenizer_path,
    })
}

fn resolve_local_dir(
    dir: &Path,
    config_override: Option<&str>,
    tokenizer_override: Option<&str>,
) -> Result<ResolvedModel> {
    // Try GGUF first (look for any .gguf file)
    if let Some(gguf) = first_file_with_ext(dir, "gguf") {
        let config_path = config_override
            .map(PathBuf::from)
            .or_else(|| {
                let p = dir.join("config.json");
                p.is_file().then_some(p)
            });
        let tokenizer_path = tokenizer_override
            .map(PathBuf::from)
            .or_else(|| {
                let p = dir.join("tokenizer.json");
                p.is_file().then_some(p)
            });
        return Ok(ResolvedModel {
            model_path: gguf,
            format: ModelFormat::Gguf,
            config_path,
            tokenizer_path,
        });
    }

    // Try SafeTensors
    let st = dir.join("model.safetensors");
    if st.is_file() {
        let config_path = config_override
            .map(PathBuf::from)
            .or_else(|| {
                let p = dir.join("config.json");
                p.is_file().then_some(p)
            });
        let tokenizer_path = tokenizer_override
            .map(PathBuf::from)
            .or_else(|| {
                let p = dir.join("tokenizer.json");
                p.is_file().then_some(p)
            });
        return Ok(ResolvedModel {
            model_path: st,
            format: ModelFormat::SafeTensors,
            config_path,
            tokenizer_path,
        });
    }

    bail!("Directory '{}' contains no .gguf or model.safetensors file", dir.display())
}

fn resolve_hf_repo(
    repo_id: &str,
    file: Option<&str>,
    config_override: Option<&str>,
    tokenizer_override: Option<&str>,
) -> Result<ResolvedModel> {
    let api = hf_hub::api::sync::Api::new().context("Failed to initialize HuggingFace API")?;
    let repo = api.model(repo_id.to_string());

    // Determine which model file to download
    let model_filename = file.unwrap_or("model.safetensors");
    let format = detect_format_from_name(model_filename)?;

    eprintln!("Downloading {} from {} ...", model_filename, repo_id);
    let model_path = repo
        .get(model_filename)
        .with_context(|| format!("Failed to download '{}' from {}", model_filename, repo_id))?;

    let config_path = if let Some(c) = config_override {
        Some(PathBuf::from(c))
    } else {
        match repo.get("config.json") {
            Ok(p) => Some(p),
            Err(_) => None,
        }
    };

    let tokenizer_path = if let Some(t) = tokenizer_override {
        Some(PathBuf::from(t))
    } else {
        match repo.get("tokenizer.json") {
            Ok(p) => Some(p),
            Err(_) => None,
        }
    };

    Ok(ResolvedModel {
        model_path,
        format,
        config_path,
        tokenizer_path,
    })
}

/// Detect format from file extension.
fn detect_format(path: &Path) -> Result<ModelFormat> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    detect_format_from_name(
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(ext),
    )
}

fn detect_format_from_name(name: &str) -> Result<ModelFormat> {
    if name.ends_with(".gguf") {
        Ok(ModelFormat::Gguf)
    } else if name.ends_with(".safetensors") {
        Ok(ModelFormat::SafeTensors)
    } else {
        bail!("Cannot detect format from '{}' — expected .gguf or .safetensors extension", name)
    }
}

/// Heuristic: contains exactly one '/' and doesn't exist on disk.
fn looks_like_hf_repo(s: &str) -> bool {
    let slash_count = s.chars().filter(|&c| c == '/').count();
    slash_count == 1 && !s.starts_with('/') && !s.starts_with('.')
}

fn first_file_with_ext(dir: &Path, ext: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.find_map(|entry| {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some(ext) {
            Some(path)
        } else {
            None
        }
    })
}

/// Auto-detect architecture from config.json: GPT-2 uses `n_embd`, Llama uses `hidden_size`.
pub fn detect_arch(config_path: &Path) -> Result<&'static str> {
    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("Failed to parse {}", config_path.display()))?;

    if value.get("n_embd").is_some() {
        Ok("gpt2")
    } else if value.get("hidden_size").is_some() {
        Ok("llama")
    } else {
        bail!(
            "Cannot auto-detect architecture from {} — no 'n_embd' (GPT-2) or 'hidden_size' (Llama) found",
            config_path.display()
        )
    }
}
