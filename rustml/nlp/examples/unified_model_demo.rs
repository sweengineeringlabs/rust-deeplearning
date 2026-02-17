//! Unified LlmModel Demo
//!
//! Demonstrates the unified model API that supports both GPT-2 and Llama architectures.
//!
//! ```bash
//! cargo run -p rustml-nlp --example unified_model_demo
//! ```

use rustml_core::Tensor;
use rustml_nlp::{
    GenerationConfig, LanguageModel, LlmModel, ModelConfig, TextGenerator,
};
use rustml_nn::{KVCache, PositionEncoding};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unified LlmModel Demo ===\n");

    // ── 1. GPT-2 style model (Learned position encoding, LayerNorm, GELU) ──
    println!("--- GPT-2 Style Model ---");
    let gpt2_config = ModelConfig {
        dim: 64,
        hidden_dim: 256,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 256,
        norm_eps: 1e-5,
        max_seq_len: 64,
        use_bias: Some(true),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
        sliding_window: None,
        attn_logit_cap: None,
        embedding_scale: None,
        rms_norm_offset: None,
        attention_bias: None,
        parallel_residual: None,
        num_local_experts: None,
        num_experts_per_tok: None,
        head_dim: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        rope_local_base_freq: None,
        rope_scaling_factor: None,
    };

    let gpt2_model = LlmModel::new(&gpt2_config)?;
    let (total, _frozen) = gpt2_model.parameter_count();
    println!("  Parameters: {}", total);
    println!("  Dim: {}, Layers: {}, Heads: {}", gpt2_config.dim, gpt2_config.n_layers, gpt2_config.n_heads);

    // Forward pass
    let input = Tensor::from_vec(
        vec![65.0, 66.0, 67.0, 68.0], // "ABCD" as byte tokens
        vec![1, 4],
    )?;
    let logits = gpt2_model.forward(&input)?;
    println!("  Input shape: {:?} -> Logits shape: {:?}", input.shape(), logits.shape());

    // ── 2. Llama style model (RoPE, RMSNorm, SwiGLU, GQA) ──
    println!("\n--- Llama Style Model ---");
    let llama_config = ModelConfig {
        dim: 64,
        hidden_dim: 172, // ~2.67x for SwiGLU
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: Some(2), // GQA: 4 Q heads, 2 KV heads
        vocab_size: 256,
        norm_eps: 1e-6,
        max_seq_len: 64,
        use_bias: Some(false),
        position_encoding: PositionEncoding::RoPE,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        chat_template: None,
        sliding_window: None,
        attn_logit_cap: None,
        embedding_scale: None,
        rms_norm_offset: None,
        attention_bias: None,
        parallel_residual: None,
        num_local_experts: None,
        num_experts_per_tok: None,
        head_dim: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        rope_local_base_freq: None,
        rope_scaling_factor: None,
    };

    let llama_model = LlmModel::new(&llama_config)?;
    let (total, _) = llama_model.parameter_count();
    println!("  Parameters: {}", total);
    println!("  Dim: {}, Layers: {}, Heads: {}, KV Heads: {}",
        llama_config.dim, llama_config.n_layers, llama_config.n_heads,
        llama_config.n_kv_heads.unwrap_or(llama_config.n_heads));

    let logits = llama_model.forward(&input)?;
    println!("  Input shape: {:?} -> Logits shape: {:?}", input.shape(), logits.shape());

    // ── 3. KV Cache inference (autoregressive decoding) ──
    println!("\n--- KV Cache Decoding ---");
    let mut cache = KVCache::new(
        llama_config.n_layers,
        llama_config.max_seq_len,
        llama_model.head_dim(),
        llama_model.num_kv_heads(),
    );

    // Prefill with prompt
    let prompt = Tensor::from_vec(vec![65.0, 66.0, 67.0, 68.0], vec![1, 4])?;
    let prefill_logits = llama_model.forward_with_cache(&prompt, &mut cache)?;
    cache.advance(4);
    println!("  Prefill: {} tokens -> logits {:?}", 4, prefill_logits.shape());

    // Decode one token at a time
    for step in 0..4 {
        let next_input = Tensor::from_vec(vec![(69 + step) as f32], vec![1, 1])?;
        let step_logits = llama_model.forward_with_cache(&next_input, &mut cache)?;
        cache.advance(1);
        println!("  Decode step {}: 1 token -> logits {:?}", step + 1, step_logits.shape());
    }

    // ── 4. Text generation via TextGenerator (works with any LanguageModel) ──
    println!("\n--- TextGenerator with LlmModel ---");
    let generator = TextGenerator::new(&gpt2_model);
    let gen_config = GenerationConfig::greedy(8);
    let output = generator.generate(&input, &gen_config)?;
    println!("  Input: 4 tokens -> Generated: {} tokens", output.shape()[1]);

    println!("\n=== Demo Complete ===");
    Ok(())
}
