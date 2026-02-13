use llmforge::core::tensor::{Tensor, DType};
use llmforge::attention::MultiHeadAttention;

#[test]
fn test_mha_forward() {
    let batch_size = 2;
    let seq_len = 10;
    let d_model = 64;
    let num_heads = 4;
    
    // Create MHA
    let mha = MultiHeadAttention::new(d_model, num_heads, false).unwrap();
    
    // Create random input: [Batch, Seq, D_model]
    let input = Tensor::new(
        vec![0u8; batch_size * seq_len * d_model * 4], 
        vec![batch_size, seq_len, d_model],
        DType::F32
    );
    
    // Forward pass
    let output = mha.forward(&input).expect("Forward pass failed");
    
    // Check output shape
    assert_eq!(output.shape(), &[batch_size, seq_len, d_model]);
    
    println!("MHA forward successful. Output shape: {:?}", output.shape());
}
