use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use crate::error::{LLMForgeError, Result};

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: Option<usize>, // Optional for GQA/MQA support
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub use_bias: Option<bool>, // Default false for Llama, true for others
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dim: 4096,
            hidden_dim: 11008,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: None,
            vocab_size: 32000,
            norm_eps: 1e-6,
            max_seq_len: 2048,
            use_bias: Some(false),
        }
    }
}

impl ModelConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: ModelConfig = serde_json::from_reader(reader)
            .map_err(|e| LLMForgeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(LLMForgeError::InvalidConfig("dim must be > 0".into()));
        }
        if self.n_heads == 0 {
            return Err(LLMForgeError::InvalidConfig("n_heads must be > 0".into()));
        }
        if self.dim % self.n_heads != 0 {
            return Err(LLMForgeError::InvalidConfig(
                format!("dim ({}) must be divisible by n_heads ({})", self.dim, self.n_heads)
            ));
        }
        if self.vocab_size == 0 {
            return Err(LLMForgeError::InvalidConfig("vocab_size must be > 0".into()));
        }
        if self.n_layers == 0 {
            return Err(LLMForgeError::InvalidConfig("n_layers must be > 0".into()));
        }
        if self.max_seq_len == 0 {
            return Err(LLMForgeError::InvalidConfig("max_seq_len must be > 0".into()));
        }
        Ok(())
    }
}

/// Runtime configuration for parallelism and thread management.
/// Must be applied (via `apply()`) before any computation to take effect.
pub struct RuntimeConfig {
    /// Number of threads for faer and rayon parallelism.
    /// 0 means auto-detect (use all available cores).
    pub num_threads: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self { num_threads: 0 }
    }
}

impl RuntimeConfig {
    /// Apply this runtime configuration globally.
    ///
    /// Sets faer's global parallelism and optionally configures
    /// rayon's global thread pool. Must be called before any
    /// computation (matmul, attention, etc.) for settings to take effect.
    pub fn apply(&self) -> Result<()> {
        use faer::{Parallelism, set_global_parallelism};

        if self.num_threads == 0 {
            set_global_parallelism(Parallelism::Rayon(0));
        } else {
            set_global_parallelism(Parallelism::Rayon(self.num_threads));
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_threads)
                .build_global()
                .map_err(|e| LLMForgeError::NotImplemented(
                    format!("Failed to set rayon thread pool: {}", e)
                ))?;
        }

        Ok(())
    }
}
