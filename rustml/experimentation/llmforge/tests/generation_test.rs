use llmforge::inference::Generator;
use llmforge::models::LlmModel;
use llmforge::tokenization::NaiveTokenizer;
use llmforge::config::ModelConfig;

#[test]
fn test_generation_basics() {
    let vocab_size = 256; 
    let tokenizer = NaiveTokenizer::new();
    
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size,
        norm_eps: 1e-5,
         max_seq_len: 128,
        use_bias: Some(true),
    };
    
    let model = LlmModel::new(&config).unwrap();
    
    let generator = Generator::new(&model, &tokenizer, 0.7);
    
    // Generate text
    let prompt = "Hi";
    // We expect basic generation to work (even if output is random garbage tokens for now)
    let output = generator.generate(prompt, 5).expect("Generation failed");
    
    println!("Generated text: {:?}", output);
    assert!(output.len() > prompt.len());
}
