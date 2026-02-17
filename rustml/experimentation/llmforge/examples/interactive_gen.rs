use llmforge::inference::Generator;
use llmforge::models::LlmModel;
use llmforge::config::{ModelConfig, PositionEncoding};
use llmforge::tokenization::NaiveTokenizer;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing LLM Forge Model...");
    
    // Model Config (Tiny for demo)
    let config = ModelConfig {
        dim: 64,
        hidden_dim: 128,
        n_layers: 4,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 256,
        norm_eps: 1e-5,
         max_seq_len: 128,
        use_bias: Some(true),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };
    
    let tokenizer = NaiveTokenizer::new();
    
    let model = LlmModel::new(&config)?;
    
    println!("Model initialized. Parameters:");
    println!("  Vocab: {}", config.vocab_size);
    println!("  Dim: {}", config.dim);
    println!("  Layers: {}", config.n_layers);
    println!("\nEntering interactive mode. Type 'quit' to exit.");
    
    let generator = Generator::new(&model, &tokenizer, 0.7);
    
    loop {
        print!("\nPrompt> ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let prompt = input.trim();
        
        if prompt == "quit" {
            break;
        }
        
        if prompt.is_empty() {
            continue;
        }
        
        println!("Generating...");
        let output = generator.generate(prompt, 20)?;
        println!("Result: {}", output);
    }
    
    Ok(())
}
