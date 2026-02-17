use llmforge::core::tensor::{Tensor, DType};
use llmforge::models::LlmModel;
use llmforge::tokenization::{Tokenizer, NaiveTokenizer};
use llmforge::config::{ModelConfig, PositionEncoding};
use bytemuck;

#[test]
fn test_end_to_end_flow() {
    let vocab_size = 256; 
    
    // 1. Initialize Tokenizer
    let tokenizer = NaiveTokenizer::new();
    
    // 2. Encode text
    let text = "Hello AI";
    let tokens = tokenizer.encode(text).expect("Encoding failed");
    println!("Encoded tokens: {:?}", tokens);
    
    // 3. Prepare Input Tensor [Batch=1, Seq=Len]
    let batch_size = 1;
    let seq_len = tokens.len();
    let mut input_data = Vec::with_capacity(seq_len);
    for &t in &tokens {
        input_data.push(t as f32);
    }
    
    let input_bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&input_data).to_vec();
    let input_tensor = Tensor::new(
        input_bytes,
        vec![batch_size, seq_len],
        DType::F32
    );
    
    // 4. Initialize Model with Config
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
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
        bos_token_id: None,
        eos_token_id: None,
        chat_template: None,
    };
    
    let model = LlmModel::new(&config).unwrap();
    
    // 5. Forward Pass
    let logits = model.forward(&input_tensor).expect("Forward failed");
    
    // Check output [1, Seq, Vocab]
    assert_eq!(logits.shape(), &[batch_size, seq_len, vocab_size]);
    println!("Logits shape: {:?}", logits.shape());
    
    // 6. Decode
    let decoded = tokenizer.decode(&tokens).expect("Decoding failed");
    assert_eq!(decoded, text);
}
