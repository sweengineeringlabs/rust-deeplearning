//! HuggingFace Hub API client with async and sync download paths.

use crate::api::error::{HubError, HubResult};
use crate::api::types::{GgufBundle, ModelBundle};
use std::path::PathBuf;

/// HuggingFace Hub API client
#[derive(Debug, Clone)]
pub struct HubApi {
    /// Base URL for the Hub
    base_url: String,
    /// Cache directory for downloaded models
    cache_dir: PathBuf,
    /// API token (optional, for private models)
    token: Option<String>,
}

impl Default for HubApi {
    fn default() -> Self {
        Self::new()
    }
}

impl HubApi {
    /// Create a new Hub API client
    pub fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("rustml")
            .join("hub");

        Self {
            base_url: "https://huggingface.co".to_string(),
            cache_dir,
            token: None,
        }
    }

    /// Create with custom cache directory
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            ..Self::new()
        }
    }

    /// Set API token for private models
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    // ======================== Async API (reqwest) ========================

    /// Download a model from HuggingFace Hub (async)
    ///
    /// # Arguments
    /// * `model_id` - Model identifier (e.g., "openai-community/gpt2")
    ///
    /// # Returns
    /// A `ModelBundle` containing paths to downloaded files
    pub async fn download_model(&self, model_id: &str) -> HubResult<ModelBundle> {
        let model_dir = self.cache_dir.join(model_id.replace('/', "--"));

        // Create cache directory if it doesn't exist
        tokio::fs::create_dir_all(&model_dir).await?;

        // Files to download for GPT-2
        let files = vec![
            "config.json",
            "model.safetensors",
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
        ];

        for file in &files {
            let file_path = model_dir.join(file);
            if !file_path.exists() {
                self.download_file(model_id, file, &file_path).await?;
            }
        }

        Ok(ModelBundle {
            model_id: model_id.to_string(),
            model_dir,
        })
    }

    /// Download a specific file from a model repository (async)
    async fn download_file(
        &self,
        model_id: &str,
        filename: &str,
        dest: &PathBuf,
    ) -> HubResult<()> {
        let url = format!(
            "{}/{}/resolve/main/{}",
            self.base_url, model_id, filename
        );

        let client = reqwest::Client::new();
        let mut request = client.get(&url);

        if let Some(ref token) = self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await.map_err(|e| {
            HubError::NetworkError(format!("Failed to download {}: {}", filename, e))
        })?;

        if !response.status().is_success() {
            // Skip optional files
            if filename == "model.safetensors" || filename == "tokenizer.json" {
                return Ok(());
            }
            return Err(HubError::NetworkError(format!(
                "Failed to download {}: HTTP {}",
                filename,
                response.status()
            )));
        }

        let bytes = response.bytes().await.map_err(|e| {
            HubError::NetworkError(format!("Failed to read response: {}", e))
        })?;

        tokio::fs::write(dest, &bytes).await?;

        Ok(())
    }

    // ======================== Sync API (hf-hub) ========================

    /// Download a model from HuggingFace Hub (synchronous, via hf-hub crate)
    ///
    /// Uses the `hf-hub` crate for synchronous downloads with automatic caching.
    /// This is the preferred path when async is not needed (e.g., CLI tools).
    pub fn download_model_sync(&self, model_id: &str) -> HubResult<ModelBundle> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| {
            HubError::NetworkError(format!("Failed to create hf-hub API: {}", e))
        })?;

        let repo = api.model(model_id.to_string());

        // Download config.json (required)
        let config_path = repo.get("config.json").map_err(|e| {
            HubError::NetworkError(format!("Failed to download config.json: {}", e))
        })?;

        let model_dir = config_path.parent().unwrap_or(&config_path).to_path_buf();

        // Try to download weights and tokenizer files (optional)
        let _weights = repo.get("model.safetensors").ok();
        let _vocab = repo.get("vocab.json").ok();
        let _merges = repo.get("merges.txt").ok();
        let _tokenizer = repo.get("tokenizer.json").ok();

        Ok(ModelBundle {
            model_id: model_id.to_string(),
            model_dir,
        })
    }

    /// Download a GGUF model file (synchronous)
    pub fn download_gguf_sync(
        &self,
        model_id: &str,
        filename: &str,
    ) -> HubResult<GgufBundle> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| {
            HubError::NetworkError(format!("Failed to create hf-hub API: {}", e))
        })?;

        let repo = api.model(model_id.to_string());
        let gguf_path = repo.get(filename).map_err(|e| {
            HubError::NetworkError(format!("Failed to download {}: {}", filename, e))
        })?;

        Ok(GgufBundle {
            gguf_path,
            model_id: Some(model_id.to_string()),
        })
    }

    // ======================== Cache management ========================

    /// Check if a model is cached locally
    pub fn is_cached(&self, model_id: &str) -> bool {
        let model_dir = self.cache_dir.join(model_id.replace('/', "--"));
        model_dir.exists() && model_dir.join("config.json").exists()
    }

    /// Get a cached model bundle without downloading
    pub fn get_cached(&self, model_id: &str) -> Option<ModelBundle> {
        if self.is_cached(model_id) {
            Some(ModelBundle {
                model_id: model_id.to_string(),
                model_dir: self.cache_dir.join(model_id.replace('/', "--")),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_api_creation() {
        let api = HubApi::new();
        assert!(!api.cache_dir().as_os_str().is_empty());
    }

    #[test]
    fn test_model_bundle_paths() {
        let bundle = ModelBundle {
            model_id: "test/model".to_string(),
            model_dir: PathBuf::from("/tmp/test"),
        };
        assert!(bundle.config_path().ends_with("config.json"));
        assert!(bundle.weights_path().ends_with("model.safetensors"));
    }
}
