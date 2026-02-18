//! GPT-2 Text Generation Example
//!
//! This example demonstrates how to:
//! 1. Download GPT-2 from HuggingFace Hub
//! 2. Load the model and tokenizer
//! 3. Generate text autoregressively
//!
//! Run with: cargo run --example gpt2_generate

use rustml_hub::HubApi;
use rustml_nlp::{
    GenerationConfig, GptConfig, GptModel, NlpResult, TextGenerator,
};
use rustml_tokenizer::BpeTokenizer;
use rustml_core::Tensor;

#[tokio::main]
async fn main() -> NlpResult<()> {
    println!("=== GPT-2 Text Generation ===\n");

    // Configuration
    let model_id = "openai-community/gpt2";
    let prompt = "The quick brown fox";
    
    println!("Model: {}", model_id);
    println!("Prompt: \"{}\"\n", prompt);

    // Step 1: Download model from HuggingFace Hub
    println!("Downloading model from HuggingFace Hub...");
    let api = HubApi::new();
    let bundle = api.download_model(model_id).await?;
    println!("Model cached at: {:?}\n", bundle.model_dir);

    // Step 2: Load configuration
    println!("Loading model configuration...");
    let config_json = bundle.load_config().await?;
    let config = GptConfig::from_hf_config(&config_json)?;
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Embedding dim: {}", config.n_embd);
    println!("  Layers: {}", config.n_layer);
    println!("  Heads: {}\n", config.n_head);

    // Step 3: Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = BpeTokenizer::from_files(
        bundle.vocab_path(),
        bundle.merges_path(),
    )?;
    println!("  Vocabulary size: {}\n", tokenizer.vocab_size());

    // Step 4: Load model weights
    println!("Loading model weights (this may take a moment)...");
    let weights = bundle.load_tensors()?;
    println!("  Loaded {} weight tensors\n", weights.len());

    // Step 5: Create model
    println!("Creating GPT-2 model...");
    let model = GptModel::from_hub_weights(config, weights)?;
    println!("  Model ready!\n");

    // Step 6: Tokenize prompt
    println!("Tokenizing prompt...");
    let input_ids = tokenizer.encode(prompt);
    println!("  Input tokens: {:?}\n", input_ids);

    // Convert to tensor
    let input_tensor = Tensor::from_vec(
        input_ids.iter().map(|&id| id as f32).collect(),
        vec![1, input_ids.len()],
    )?;

    // Step 7: Generate text
    println!("Generating text...");
    let generator = TextGenerator::new(&model);
    
    // Try different sampling strategies
    let configs = vec![
        ("Greedy", GenerationConfig::greedy(30)),
        ("Temperature 0.7", GenerationConfig::with_temperature(30, 0.7)),
        ("Top-k 50", GenerationConfig::with_top_k(30, 50, 0.8)),
        ("Top-p 0.9", GenerationConfig::with_top_p(30, 0.9, 0.8)),
    ];

    for (name, gen_config) in configs {
        println!("\n--- {} ---", name);
        
        let output = generator.generate(&input_tensor, &gen_config)?;
        let output_ids: Vec<u32> = output.iter().map(|f| f as u32).collect();
        let generated_text = tokenizer.decode(&output_ids);
        
        println!("Generated: {}", generated_text);
    }

    println!("\n=== Generation Complete ===");

    Ok(())
}
