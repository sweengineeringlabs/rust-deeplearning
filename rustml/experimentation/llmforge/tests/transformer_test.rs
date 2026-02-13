use llmforge::core::tensor::{Tensor, DType};
use llmforge::config::PositionEncoding;
use llmforge::transformer::TransformerBlock;

#[test]
fn test_transformer_block_forward() {
    let batch_size = 2;
    let seq_len = 8;
    let d_model = 32;
    let num_heads = 4;
    let hidden_dim = 64;

    // Create Transformer Block
    let block = TransformerBlock::new(
        d_model,
        num_heads,
        None,
        hidden_dim,
        true, // bias
        1e-5, // eps
        false,
        PositionEncoding::Learned,
        128,
        10000.0,
    ).unwrap();
    
    // Create random input: [Batch, Seq, D_model]
    let input = Tensor::new(
        vec![0u8; batch_size * seq_len * d_model * 4], 
        vec![batch_size, seq_len, d_model],
        DType::F32
    );
    
    // Forward pass
    let output = block.forward(&input).expect("Forward pass failed");
    
    // Check output shape
    assert_eq!(output.shape(), &[batch_size, seq_len, d_model]);
    
    println!("Transformer Block forward successful. Output shape: {:?}", output.shape());
}
