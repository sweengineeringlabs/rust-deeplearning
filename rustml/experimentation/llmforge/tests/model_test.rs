use llmforge::core::tensor::{Tensor, DType};
use llmforge::models::LlmModel;
use llmforge::config::{ModelConfig, PositionEncoding};

#[test]
fn test_lldm_model_forward() {
    let batch_size = 2;
    let seq_len = 8;
    let vocab_size = 100;
    
    // Create Config
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
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
    
    // Create Model
    let model = LlmModel::new(&config).unwrap();
    
    // Create dummy input (indices)
    // Using F32 tensor for indices as per current limitation
    let input = Tensor::new(
        vec![0u8; batch_size * seq_len * 4], 
        vec![batch_size, seq_len],
        DType::F32 
    );
    
    // Result<Tensor>
    let output = model.forward(&input);
    
    match output {
        Ok(out) => {
             assert_eq!(out.shape(), &[batch_size, seq_len, vocab_size]);
             println!("Model forward successful. Output shape: {:?}", out.shape());
        },
        Err(e) => {
            println!("Model forward failed: {:?}", e);
            panic!("Model forward failed: {:?}", e);
        }
    }
}
