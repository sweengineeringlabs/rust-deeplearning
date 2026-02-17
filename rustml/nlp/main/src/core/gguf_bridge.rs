//! Bridge between rustml-gguf types and rustml-core/nlp types.
//!
//! Provides conversions from `LoadedTensor` → `Tensor` and
//! `GgufModelConfig` → `ModelConfig` for loading GGUF models.

use crate::api::error::NlpResult;
use crate::api::types::ModelConfig;
use rustml_core::{DType, Tensor};
use rustml_gguf::{GgufModelConfig, LoadedDType, LoadedTensor};
use rustml_nn::PositionEncoding;
use std::collections::HashMap;

/// Convert a LoadedDType (GGUF) to DType (core).
pub fn loaded_dtype_to_dtype(dtype: LoadedDType) -> DType {
    match dtype {
        LoadedDType::F32 => DType::F32,
        LoadedDType::F16 => DType::F16,
        LoadedDType::Q8_0 => DType::Q8_0,
        LoadedDType::Q4_0 => DType::Q4_0,
    }
}

/// Convert a LoadedTensor (GGUF) to a Tensor (core).
pub fn loaded_tensor_to_tensor(lt: LoadedTensor) -> Tensor {
    let dtype = loaded_dtype_to_dtype(lt.dtype);
    Tensor::new(lt.data, lt.shape, dtype)
}

/// Convert a map of LoadedTensors to a map of Tensors.
pub fn convert_tensors(tensors: HashMap<String, LoadedTensor>) -> HashMap<String, Tensor> {
    tensors
        .into_iter()
        .map(|(name, lt)| (name, loaded_tensor_to_tensor(lt)))
        .collect()
}

/// Convert a GgufModelConfig to a ModelConfig suitable for model construction.
///
/// For Gemma 3 models, sets architecture-specific defaults:
/// - `sliding_window_pattern = 6` (every 6th layer is global attention)
/// - `query_pre_attn_scalar = head_dim` (custom attention scaling)
/// - `rope_local_base_freq = 10000.0` (local layer RoPE theta)
/// - `rms_norm_offset = 0.0` (GGUF converter already bakes in the +1.0 shift)
/// - `embedding_scale = sqrt(dim)` (Gemma embedding scaling)
pub fn gguf_config_to_model_config(gc: &GgufModelConfig) -> NlpResult<ModelConfig> {
    let is_gemma3 = gc.architecture == "gemma3";
    let head_dim = gc.head_dim.unwrap_or(gc.dim / gc.n_heads);

    let config = ModelConfig {
        dim: gc.dim,
        hidden_dim: gc.hidden_dim,
        n_layers: gc.n_layers,
        n_heads: gc.n_heads,
        n_kv_heads: gc.n_kv_heads,
        vocab_size: gc.vocab_size,
        norm_eps: gc.norm_eps,
        max_seq_len: gc.max_seq_len,
        use_bias: Some(false),
        position_encoding: PositionEncoding::RoPE,
        causal: true,
        rope_theta: gc.rope_theta,
        bos_token_id: gc.bos_token_id,
        eos_token_id: gc.eos_token_id,
        chat_template: gc.chat_template.clone(),
        sliding_window: gc.sliding_window,
        attn_logit_cap: None,
        embedding_scale: if is_gemma3 { Some((gc.dim as f32).sqrt()) } else { None },
        // GGUF converter (convert_hf_to_gguf.py) already adds +1.0 to Gemma norm weights,
        // so we must NOT add it again at forward time. Explicitly set 0.0 for Gemma.
        rms_norm_offset: if is_gemma3 { Some(0.0) } else { None },
        attention_bias: None,
        parallel_residual: None,
        num_local_experts: None,
        num_experts_per_tok: None,
        head_dim: gc.head_dim,
        sliding_window_pattern: if is_gemma3 { Some(6) } else { None },
        query_pre_attn_scalar: if is_gemma3 { Some(head_dim as f32) } else { None },
        rope_local_base_freq: if is_gemma3 { Some(10000.0) } else { None },
        rope_scaling_factor: None,
    };
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustml_core::f32_vec_to_bytes;

    #[test]
    fn test_loaded_tensor_to_tensor_f32() {
        let data = f32_vec_to_bytes(vec![1.0, 2.0, 3.0, 4.0]);
        let lt = LoadedTensor {
            data,
            shape: vec![2, 2],
            dtype: LoadedDType::F32,
        };
        let t = loaded_tensor_to_tensor(lt);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_convert_tensors_map() {
        let data = f32_vec_to_bytes(vec![1.0, 2.0]);
        let mut loaded = HashMap::new();
        loaded.insert(
            "weight".to_string(),
            LoadedTensor {
                data,
                shape: vec![2],
                dtype: LoadedDType::F32,
            },
        );
        let tensors = convert_tensors(loaded);
        assert!(tensors.contains_key("weight"));
        assert_eq!(tensors["weight"].shape(), &[2]);
    }

    #[test]
    fn test_gguf_config_to_model_config_gemma3() {
        let gc = GgufModelConfig {
            architecture: "gemma3".to_string(),
            dim: 1152,
            hidden_dim: 6912,
            n_layers: 26,
            n_heads: 4,
            n_kv_heads: Some(1),
            vocab_size: 262144,
            norm_eps: 1e-6,
            max_seq_len: 32768,
            rope_theta: 1000000.0,
            bos_token_id: Some(2),
            eos_token_id: Some(1),
            chat_template: None,
            head_dim: Some(256),
            sliding_window: Some(512),
        };
        let mc = gguf_config_to_model_config(&gc).unwrap();
        assert_eq!(mc.dim, 1152);
        assert_eq!(mc.head_dim, Some(256));
        assert_eq!(mc.sliding_window, Some(512));
        assert_eq!(mc.sliding_window_pattern, Some(6));
        assert_eq!(mc.query_pre_attn_scalar, Some(256.0));
        assert_eq!(mc.rope_local_base_freq, Some(10000.0));
        assert_eq!(mc.rms_norm_offset, Some(0.0)); // GGUF already has +1.0 baked in
        assert!((mc.embedding_scale.unwrap() - (1152.0_f32).sqrt()).abs() < 0.01);
        assert_eq!(mc.rope_theta, 1000000.0);
    }

    #[test]
    fn test_gguf_config_to_model_config_llama() {
        let gc = GgufModelConfig {
            architecture: "llama".to_string(),
            dim: 4096,
            hidden_dim: 11008,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: Some(8),
            vocab_size: 32000,
            norm_eps: 1e-5,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            chat_template: None,
            head_dim: None,
            sliding_window: None,
        };
        let mc = gguf_config_to_model_config(&gc).unwrap();
        assert_eq!(mc.dim, 4096);
        assert_eq!(mc.head_dim, None);
        assert_eq!(mc.sliding_window_pattern, None);
        assert_eq!(mc.embedding_scale, None);
        assert_eq!(mc.rms_norm_offset, None);
    }
}
