//! Data types for hub API operations

use crate::api::error::{HubError, HubResult};
use std::path::PathBuf;

/// A bundle of downloaded model files (SafeTensors format)
#[derive(Debug, Clone)]
pub struct ModelBundle {
    /// Model identifier
    pub model_id: String,
    /// Path to the model directory
    pub model_dir: PathBuf,
}

impl ModelBundle {
    /// Get path to config.json
    pub fn config_path(&self) -> PathBuf {
        self.model_dir.join("config.json")
    }

    /// Get path to model weights (SafeTensors format)
    pub fn weights_path(&self) -> PathBuf {
        self.model_dir.join("model.safetensors")
    }

    /// Get path to vocab.json
    pub fn vocab_path(&self) -> PathBuf {
        self.model_dir.join("vocab.json")
    }

    /// Get path to merges.txt
    pub fn merges_path(&self) -> PathBuf {
        self.model_dir.join("merges.txt")
    }

    /// Get path to tokenizer.json (HuggingFace universal tokenizer)
    pub fn tokenizer_json_path(&self) -> PathBuf {
        self.model_dir.join("tokenizer.json")
    }

    /// Load model configuration
    pub async fn load_config(&self) -> HubResult<serde_json::Value> {
        let content = tokio::fs::read_to_string(self.config_path()).await?;
        serde_json::from_str(&content).map_err(|e| HubError::ParseError(e.to_string()))
    }

    /// Load model configuration synchronously
    pub fn load_config_sync(&self) -> HubResult<serde_json::Value> {
        let content = std::fs::read_to_string(self.config_path())?;
        serde_json::from_str(&content).map_err(|e| HubError::ParseError(e.to_string()))
    }

    /// Load tensors from the model (converts all to F32)
    pub fn load_tensors(&self) -> HubResult<std::collections::HashMap<String, rustml_core::Tensor>> {
        let loader = crate::core::safetensors::SafeTensorLoader::new();
        loader.load(&self.weights_path())
    }

    /// Load tensors keeping original dtype via mmap (zero-copy)
    pub fn load_tensors_mmap(&self) -> HubResult<std::collections::HashMap<String, rustml_core::Tensor>> {
        crate::core::safetensors::load_safetensors_mmap(&self.weights_path())
    }
}

/// A bundle for GGUF model files
#[derive(Debug, Clone)]
pub struct GgufBundle {
    /// Path to the GGUF file
    pub gguf_path: PathBuf,
    /// Model identifier (if downloaded from hub)
    pub model_id: Option<String>,
}

impl GgufBundle {
    /// Create from a local GGUF file path
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            gguf_path: path.into(),
            model_id: None,
        }
    }
}
