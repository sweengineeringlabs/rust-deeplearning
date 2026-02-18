//! Gemma 3 Model Loading Example
//!
//! Demonstrates how to:
//! 1. Download Gemma 3 4B from HuggingFace Hub
//! 2. Remap weights using WeightMap::gemma3()
//! 3. Build the model with from_pretrained_gemma3()
//! 4. Run a forward pass and generate text
//!
//! Requires a HuggingFace token with access to google/gemma-3-4b-pt.
//!
//! ```bash
//! HF_TOKEN=hf_... cargo run -p rustml-nlp --example gemma3_load
//! ```

use rustml_core::Tensor;
use rustml_hub::HubApi;
use rustml_nlp::{
    GenerationConfig, LanguageModel, LlmModel, ModelConfig, TextGenerator,
    WeightMap,
};
use rustml_tokenizer::{HFTokenizer, Tokenizer};
use rustml_nn::KVCache;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Gemma 3 Model Loading ===\n");

    let model_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "google/gemma-3-4b-pt".to_string());
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The capital of France is".to_string());

    println!("Model: {}", model_id);
    println!("Prompt: \"{}\"\n", prompt);

    // Step 1: Download model from HuggingFace Hub
    println!("Downloading model from HuggingFace Hub...");
    let api = if let Ok(token) = std::env::var("HF_TOKEN") {
        HubApi::new().with_token(token)
    } else {
        eprintln!("Warning: HF_TOKEN not set. Gemma 3 requires authentication.");
        HubApi::new()
    };
    let bundle = api.download_model(&model_id).await?;
    println!("  Cached at: {:?}\n", bundle.model_dir);

    // Step 2: Load configuration
    println!("Loading configuration...");
    let config_json = bundle.load_config().await?;
    let config = ModelConfig::from_json_value(&config_json)?;
    println!("  dim: {}", config.dim);
    println!("  hidden_dim: {}", config.hidden_dim);
    println!("  layers: {}", config.n_layers);
    println!("  heads: {}, kv_heads: {:?}", config.n_heads, config.n_kv_heads);
    println!("  head_dim: {:?}", config.head_dim);
    println!("  vocab_size: {}", config.vocab_size);
    println!("  sliding_window: {:?}", config.sliding_window);
    println!("  sliding_window_pattern: {:?}", config.sliding_window_pattern);
    println!("  rope_theta: {}", config.rope_theta);
    println!("  rope_local_base_freq: {:?}", config.rope_local_base_freq);
    println!("  rope_scaling_factor: {:?}", config.rope_scaling_factor);
    println!("  query_pre_attn_scalar: {:?}", config.query_pre_attn_scalar);
    println!();

    // Step 3: Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = HFTokenizer::from_file(bundle.model_dir.join("tokenizer.json"))?;
    println!("  Tokenizer ready\n");

    // Step 4: Load and remap weights
    println!("Loading SafeTensors weights (this may take a while for 4B)...");
    let hf_weights = bundle.load_tensors_mmap()?;
    println!("  Loaded {} HF weight tensors", hf_weights.len());

    let weight_map = WeightMap::gemma3(config.n_layers);
    let weights = weight_map.remap(hf_weights);
    println!("  Remapped to {} internal tensors\n", weights.len());

    // Step 5: Build model
    println!("Building Gemma 3 model...");
    let model = LlmModel::from_pretrained_gemma3(&config, weights)?;
    let (total, _) = model.parameter_count();
    println!("  Parameters: {}", total);
    println!("  Model ready!\n");

    // Step 6: Tokenize
    println!("Tokenizing prompt...");
    let input_ids = tokenizer.encode(&prompt)?;
    println!("  Tokens: {:?}\n", input_ids);

    let input_tensor = Tensor::from_vec(
        input_ids.iter().map(|id| *id as f32).collect(),
        vec![1, input_ids.len()],
    )?;

    // Step 7: Forward pass
    println!("Running forward pass...");
    let logits = model.forward(&input_tensor)?;
    println!("  Logits shape: {:?}", logits.shape());

    // Step 8: Generate with KV cache
    println!("\nGenerating text (greedy, 32 tokens)...");
    let generator = TextGenerator::new(&model);
    let gen_config = GenerationConfig::greedy(32);
    let output = generator.generate(&input_tensor, &gen_config)?;
    let output_ids: Vec<u32> = output.iter().map(|f| f as u32).collect();
    let generated_text = tokenizer.decode(&output_ids)?;
    println!("  Generated: {}", generated_text);

    // Step 9: KV cache decoding demo
    println!("\n--- KV Cache Decoding ---");
    let mut cache = KVCache::new(
        config.n_layers,
        config.max_seq_len,
        model.head_dim(),
        model.num_kv_heads(),
    );

    let prefill_logits = model.forward_with_cache(&input_tensor, &mut cache)?;
    cache.advance(input_ids.len());
    println!("  Prefill: {} tokens -> logits {:?}", input_ids.len(), prefill_logits.shape());

    // Decode a few tokens
    let last_logits_data: Vec<f32> = prefill_logits.iter().collect();
    let vocab = config.vocab_size;
    let last_token_logits = &last_logits_data[last_logits_data.len() - vocab..];
    let next_token = last_token_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    for step in 0..4 {
        let tok = if step == 0 { next_token } else { next_token };
        let next_input = Tensor::from_vec(vec![tok as f32], vec![1, 1])?;
        let step_logits = model.forward_with_cache(&next_input, &mut cache)?;
        cache.advance(1);
        println!("  Decode step {}: token {} -> logits {:?}", step + 1, tok, step_logits.shape());
    }

    println!("\n=== Done ===");
    Ok(())
}
