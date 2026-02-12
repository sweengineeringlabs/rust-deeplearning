//! Public configuration types for NLP models

use crate::api::error::NlpResult;

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
