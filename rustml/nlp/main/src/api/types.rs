//! Public configuration types for NLP models

use crate::api::error::{NlpError, NlpResult};
use rustml_core::Tensor;
use rustml_nn::{KVCache, PositionEncoding};

/// GPT-2 Model Configuration
///
/// Supports all GPT-2 variants:
/// - Small (124M): 768 embed, 12 layers, 12 heads
/// - Medium (355M): 1024 embed, 24 layers, 16 heads
/// - Large (774M): 1280 embed, 36 layers, 20 heads
/// - XL (1.5B): 1600 embed, 48 layers, 25 heads
#[derive(Debug, Clone)]
pub struct GptConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length (context window)
    pub n_positions: usize,
    /// Embedding dimension
    pub n_embd: usize,
    /// Number of transformer layers
    pub n_layer: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
}

impl GptConfig {
    /// GPT-2 Small (124M parameters)
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            layer_norm_eps: 1e-5,
        }
    }

    /// GPT-2 Medium (355M parameters)
    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            layer_norm_eps: 1e-5,
        }
    }

    /// GPT-2 Large (774M parameters)
    pub fn gpt2_large() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1280,
            n_layer: 36,
            n_head: 20,
            layer_norm_eps: 1e-5,
        }
    }

    /// GPT-2 XL (1.5B parameters)
    pub fn gpt2_xl() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1600,
            n_layer: 48,
            n_head: 25,
            layer_norm_eps: 1e-5,
        }
    }

    /// Create config from HuggingFace config.json
    pub fn from_hf_config(config: &serde_json::Value) -> NlpResult<Self> {
        Ok(Self {
            vocab_size: config["vocab_size"].as_u64().unwrap_or(50257) as usize,
            n_positions: config["n_positions"].as_u64().unwrap_or(1024) as usize,
            n_embd: config["n_embd"].as_u64().unwrap_or(768) as usize,
            n_layer: config["n_layer"].as_u64().unwrap_or(12) as usize,
            n_head: config["n_head"].as_u64().unwrap_or(12) as usize,
            layer_norm_eps: config["layer_norm_epsilon"]
                .as_f64()
                .unwrap_or(1e-5) as f32,
        })
    }
}

impl Default for GptConfig {
    fn default() -> Self {
        Self::gpt2_small()
    }
}

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling (1.0 = normal, less than 1.0 = more deterministic, greater than 1.0 = more random)
    pub temperature: f32,
    /// Top-k sampling: keep only top k tokens
    pub top_k: Option<usize>,
    /// Top-p (nucleus) sampling: keep tokens with cumulative probability less than or equal to p
    pub top_p: Option<f32>,
    /// Whether to use greedy decoding (overrides temperature and sampling)
    pub do_sample: bool,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
    /// End-of-sequence token ID
    pub eos_token_id: Option<u32>,
    /// Pad token ID
    pub pad_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 50,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            do_sample: true,
            repetition_penalty: 1.0,
            eos_token_id: Some(50256), // GPT-2 EOS token
            pad_token_id: Some(50256),
        }
    }
}

impl GenerationConfig {
    /// Create a greedy decoding config
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            do_sample: false,
            ..Default::default()
        }
    }

    /// Create a config with temperature sampling
    pub fn with_temperature(max_new_tokens: usize, temperature: f32) -> Self {
        Self {
            max_new_tokens,
            temperature,
            do_sample: true,
            ..Default::default()
        }
    }

    /// Create a config with top-k sampling
    pub fn with_top_k(max_new_tokens: usize, top_k: usize, temperature: f32) -> Self {
        Self {
            max_new_tokens,
            temperature,
            top_k: Some(top_k),
            do_sample: true,
            ..Default::default()
        }
    }

    /// Create a config with nucleus (top-p) sampling
    pub fn with_top_p(max_new_tokens: usize, top_p: f32, temperature: f32) -> Self {
        Self {
            max_new_tokens,
            temperature,
            top_p: Some(top_p),
            do_sample: true,
            ..Default::default()
        }
    }
}

// ======================== Unified Model Config ========================

/// Unified model configuration supporting GPT-2, Llama, and future architectures.
///
/// This is a superset config — GPT-2 uses `position_encoding = Learned` with bias,
/// while Llama uses `position_encoding = RoPE` without bias.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    /// Model dimension (hidden size)
    pub dim: usize,
    /// Feed-forward hidden dimension (intermediate size)
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of KV heads (None = same as n_heads, for GQA/MQA)
    pub n_kv_heads: Option<usize>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Normalization epsilon
    pub norm_eps: f32,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Whether to use bias in linear layers (default false for Llama, true for GPT-2)
    pub use_bias: Option<bool>,
    /// Position encoding strategy
    #[serde(default = "default_position_encoding")]
    pub position_encoding: PositionEncoding,
    /// Whether to use causal masking
    #[serde(default = "default_causal")]
    pub causal: bool,
    /// RoPE theta parameter
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Beginning-of-sequence token ID
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ID
    #[serde(default)]
    pub eos_token_id: Option<u32>,
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
            bos_token_id: None,
            eos_token_id: None,
        }
    }
}

/// HuggingFace Llama config.json format (internal).
#[derive(serde::Deserialize)]
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

/// HuggingFace GPT-2 config.json format (internal).
#[derive(serde::Deserialize)]
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
    /// Load from a HuggingFace Llama config.json file.
    pub fn from_hf_llama<P: AsRef<std::path::Path>>(path: P) -> NlpResult<Self> {
        let file = std::fs::File::open(&path)?;
        let reader = std::io::BufReader::new(file);
        let hf: HFLlamaConfig = serde_json::from_reader(reader)
            .map_err(|e| NlpError::ModelError(format!("Invalid Llama config: {}", e)))?;

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
            bos_token_id: Some(1),
            eos_token_id: Some(2),
        };
        config.validate()?;
        Ok(config)
    }

    /// Load from a HuggingFace GPT-2 config.json file.
    pub fn from_hf_gpt2<P: AsRef<std::path::Path>>(path: P) -> NlpResult<Self> {
        let file = std::fs::File::open(&path)?;
        let reader = std::io::BufReader::new(file);
        let hf: HFGpt2Config = serde_json::from_reader(reader)
            .map_err(|e| NlpError::ModelError(format!("Invalid GPT-2 config: {}", e)))?;

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
            bos_token_id: None,
            eos_token_id: Some(50256),
        };
        config.validate()?;
        Ok(config)
    }

    /// Load from a serde_json::Value (parsed config.json).
    pub fn from_json_value(config: &serde_json::Value) -> NlpResult<Self> {
        // Auto-detect architecture
        let model_type = config["model_type"].as_str().unwrap_or("");
        match model_type {
            "llama" => {
                let hf: HFLlamaConfig = serde_json::from_value(config.clone())
                    .map_err(|e| NlpError::ModelError(format!("Invalid Llama config: {}", e)))?;
                let c = Self {
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
                    bos_token_id: Some(1),
                    eos_token_id: Some(2),
                };
                c.validate()?;
                Ok(c)
            }
            _ => {
                // Default to GPT-2 style
                let hf: HFGpt2Config = serde_json::from_value(config.clone())
                    .map_err(|e| NlpError::ModelError(format!("Invalid GPT-2 config: {}", e)))?;
                let c = Self {
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
                    bos_token_id: None,
                    eos_token_id: Some(50256),
                };
                c.validate()?;
                Ok(c)
            }
        }
    }

    /// Validate configuration constraints.
    pub fn validate(&self) -> NlpResult<()> {
        if self.dim == 0 {
            return Err(NlpError::ModelError("dim must be > 0".into()));
        }
        if self.n_heads == 0 {
            return Err(NlpError::ModelError("n_heads must be > 0".into()));
        }
        if self.dim % self.n_heads != 0 {
            return Err(NlpError::ModelError(format!(
                "dim ({}) must be divisible by n_heads ({})", self.dim, self.n_heads
            )));
        }
        if self.vocab_size == 0 {
            return Err(NlpError::ModelError("vocab_size must be > 0".into()));
        }
        if self.n_layers == 0 {
            return Err(NlpError::ModelError("n_layers must be > 0".into()));
        }
        if let Some(n_kv) = self.n_kv_heads {
            if n_kv == 0 {
                return Err(NlpError::ModelError("n_kv_heads must be > 0".into()));
            }
            if self.n_heads % n_kv != 0 {
                return Err(NlpError::ModelError(format!(
                    "n_heads ({}) must be divisible by n_kv_heads ({})", self.n_heads, n_kv
                )));
            }
        }
        Ok(())
    }
}

// ======================== LanguageModel trait ========================

/// Trait for language models that can be used with TextGenerator.
pub trait LanguageModel {
    /// Forward pass: input_ids [B, S] -> logits [B, S, vocab_size]
    fn forward(&self, input_ids: &Tensor) -> NlpResult<Tensor>;

    /// Forward pass with KV cache for autoregressive decoding.
    fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KVCache) -> NlpResult<Tensor>;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Maximum sequence length (context window).
    fn max_sequence_length(&self) -> usize;

    /// Model dimension.
    fn embedding_dim(&self) -> usize;

    /// Number of transformer layers.
    fn num_layers(&self) -> usize;

    /// Number of KV heads (for cache sizing).
    fn num_kv_heads(&self) -> usize;

    /// Head dimension (for cache sizing).
    fn head_dim(&self) -> usize;
}
