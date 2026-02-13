use llmforge::inference::Generator;
use llmforge::models::LlmModel;
use llmforge::tokenization::NaiveTokenizer;
use llmforge::config::{ModelConfig, PositionEncoding};

fn tiny_model_and_tokenizer() -> (LlmModel, NaiveTokenizer) {
    let config = ModelConfig {
        dim: 32,
        hidden_dim: 64,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: None,
        vocab_size: 256,
        norm_eps: 1e-5,
        max_seq_len: 128,
        use_bias: Some(true),
        position_encoding: PositionEncoding::Learned,
        causal: true,
        rope_theta: 10000.0,
    };
    let model = LlmModel::new(&config).unwrap();
    let tokenizer = NaiveTokenizer::new();
    (model, tokenizer)
}

#[test]
fn test_generation_basics() {
    let (model, tokenizer) = tiny_model_and_tokenizer();
    let generator = Generator::new(&model, &tokenizer, 0.7);

    let prompt = "Hi";
    let output = generator.generate(prompt, 5).expect("Generation failed");
    println!("Generated text: {:?}", output);
    assert!(output.len() > prompt.len());
}

#[test]
fn test_generation_with_top_k() {
    let (model, tokenizer) = tiny_model_and_tokenizer();
    let generator = Generator::new(&model, &tokenizer, 0.8)
        .with_top_k(5);

    let output = generator.generate("A", 5).expect("Generation with top_k failed");
    assert!(output.len() > 1);
}

#[test]
fn test_generation_with_top_p() {
    let (model, tokenizer) = tiny_model_and_tokenizer();
    let generator = Generator::new(&model, &tokenizer, 0.8)
        .with_top_p(0.9);

    let output = generator.generate("A", 5).expect("Generation with top_p failed");
    assert!(output.len() > 1);
}

#[test]
fn test_generation_with_repetition_penalty() {
    let (model, tokenizer) = tiny_model_and_tokenizer();
    let generator = Generator::new(&model, &tokenizer, 0.8)
        .with_repetition_penalty(1.2);

    let output = generator.generate("A", 5).expect("Generation with rep_penalty failed");
    assert!(output.len() > 1);
}

#[test]
fn test_generation_with_all_sampling_options() {
    let (model, tokenizer) = tiny_model_and_tokenizer();
    let generator = Generator::new(&model, &tokenizer, 0.8)
        .with_top_k(10)
        .with_top_p(0.95)
        .with_repetition_penalty(1.1);

    let output = generator.generate("AB", 5).expect("Combined sampling failed");
    assert!(output.len() > 2);
}

#[test]
fn test_streaming_generation() {
    let (model, tokenizer) = tiny_model_and_tokenizer();
    let generator = Generator::new(&model, &tokenizer, 0.7);

    let mut streamed_tokens = Vec::new();
    let output = generator.generate_stream("Hi", 5, |token| {
        streamed_tokens.push(token);
        true
    }).expect("Streaming generation failed");

    assert!(!streamed_tokens.is_empty(), "Callback should receive tokens");
    assert!(output.len() > 2);
}

#[test]
fn test_streaming_early_stop() {
    let (model, tokenizer) = tiny_model_and_tokenizer();
    let generator = Generator::new(&model, &tokenizer, 0.7);

    let mut count = 0;
    let _output = generator.generate_stream("Hi", 20, |_token| {
        count += 1;
        count < 3 // stop after 3 tokens
    }).expect("Streaming early stop failed");

    assert!(count <= 3, "Should have stopped after 3 callbacks, got {}", count);
}

#[test]
fn test_beam_search() {
    let (model, tokenizer) = tiny_model_and_tokenizer();
    let generator = Generator::new(&model, &tokenizer, 0.0);

    let output = generator.generate_beam("Hi", 5, 3)
        .expect("Beam search failed");
    assert!(output.len() > 2);
}

#[test]
fn test_batch_generation() {
    let (model, tokenizer) = tiny_model_and_tokenizer();
    let generator = Generator::new(&model, &tokenizer, 0.7);

    let prompts = vec!["A", "B", "C"];
    let outputs = generator.generate_batch(&prompts, 3)
        .expect("Batch generation failed");

    assert_eq!(outputs.len(), 3);
    for output in &outputs {
        assert!(output.len() >= 1);
    }
}
