use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use crate::error::{LLMForgeError, Result};

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PositionEncoding {
    Learned,
    #[serde(rename = "rope")]
    RoPE,
    #[serde(rename = "alibi")]
    ALiBi,
    None,
}

impl Default for PositionEncoding {
    fn default() -> Self {
        PositionEncoding::Learned
    }
}

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
    #[serde(default = "default_position_encoding")]
    pub position_encoding: PositionEncoding,
    #[serde(default = "default_causal")]
    pub causal: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_position_encoding() -> PositionEncoding { PositionEncoding::Learned }
fn default_causal() -> bool { true }
fn default_rope_theta() -> f32 { 10000.0 }

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
            position_encoding: PositionEncoding::Learned,
            causal: true,
            rope_theta: 10000.0,
        }
    }
}

/// HuggingFace Llama-2 config.json format.
#[derive(Deserialize)]
struct HFLlamaConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f32,
    max_position_embeddings: usize,
    rope_theta: Option<f32>,
}

/// HuggingFace GPT-2 config.json format.
#[derive(Deserialize)]
struct HFGpt2Config {
    n_embd: usize,
    n_inner: Option<usize>,
    n_layer: usize,
    n_head: usize,
    vocab_size: usize,
    n_positions: usize,
    layer_norm_epsilon: Option<f32>,
}

impl ModelConfig {
    /// Load a ModelConfig from a HuggingFace Llama-2 config.json file.
    pub fn from_hf_llama2<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let hf: HFLlamaConfig = serde_json::from_reader(reader)
            .map_err(|e| LLMForgeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;

        let config = Self {
            dim: hf.hidden_size,
            hidden_dim: hf.intermediate_size,
            n_layers: hf.num_hidden_layers,
            n_heads: hf.num_attention_heads,
            n_kv_heads: hf.num_key_value_heads,
            vocab_size: hf.vocab_size,
            norm_eps: hf.rms_norm_eps,
            max_seq_len: hf.max_position_embeddings,
            use_bias: Some(false),
            position_encoding: PositionEncoding::RoPE,
            causal: true,
            rope_theta: hf.rope_theta.unwrap_or(10000.0),
        };
        config.validate()?;
        Ok(config)
    }

    /// Load a ModelConfig from a HuggingFace GPT-2 config.json file.
    pub fn from_hf_gpt2<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let hf: HFGpt2Config = serde_json::from_reader(reader)
            .map_err(|e| LLMForgeError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;

        let config = Self {
            dim: hf.n_embd,
            hidden_dim: hf.n_inner.unwrap_or(4 * hf.n_embd),
            n_layers: hf.n_layer,
            n_heads: hf.n_head,
            n_kv_heads: None,
            vocab_size: hf.vocab_size,
            norm_eps: hf.layer_norm_epsilon.unwrap_or(1e-5),
            max_seq_len: hf.n_positions,
            use_bias: Some(true),
            position_encoding: PositionEncoding::Learned,
            causal: true,
            rope_theta: 10000.0,
        };
        config.validate()?;
        Ok(config)
    }

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
        if let Some(n_kv) = self.n_kv_heads {
            if n_kv == 0 {
                return Err(LLMForgeError::InvalidConfig("n_kv_heads must be > 0".into()));
            }
            if self.n_heads % n_kv != 0 {
                return Err(LLMForgeError::InvalidConfig(
                    format!("n_heads ({}) must be divisible by n_kv_heads ({})", self.n_heads, n_kv)
                ));
            }
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
