# RustML - Machine Learning in Pure Rust

A pure Rust machine learning library focused on transformer models, starting with GPT-2.

## 🎯 Features

- **GPT-2 Compatible**: Load and run pre-trained GPT-2 models from HuggingFace
- **Pure Rust**: No external ML dependencies, everything implemented from scratch
- **Text Generation**: Support for temperature, top-k, and top-p sampling
- **HuggingFace Hub**: Download models directly from HuggingFace

## 📦 Crates

| Crate | Description |
|-------|-------------|
| `rustml-core` | Core tensor operations |
| `rustml-nn` | Neural network layers (Linear, Embedding, LayerNorm, Attention) |
| `rustml-hub` | HuggingFace Hub integration and SafeTensors loading |
| `rustml-nlp` | NLP models (GPT-2) and text generation |

## 🚀 Quick Start

```rust
use rustml_nlp::{GptModel, GptConfig, TextGenerator, GenerationConfig};
use rustml_nlp::tokenizer::BpeTokenizer;
use rustml_hub::HubApi;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Download GPT-2 from HuggingFace
    let api = HubApi::new();
    let bundle = api.download_model("openai-community/gpt2").await?;
    
    // Load model
    let config = GptConfig::gpt2_small();
    let weights = bundle.load_tensors()?;
    let model = GptModel::from_hub_weights(config, weights)?;
    
    // Load tokenizer
    let tokenizer = BpeTokenizer::from_files(
        bundle.vocab_path(),
        bundle.merges_path(),
    )?;
    
    // Generate text
    let input_ids = tokenizer.encode("The quick brown fox");
    let input_tensor = Tensor::from_vec(
        input_ids.iter().map(|&id| id as f32).collect(),
        vec![1, input_ids.len()],
    )?;
    
    let generator = TextGenerator::new(&model);
    let output = generator.generate(&input_tensor, &GenerationConfig {
        max_new_tokens: 50,
        temperature: 0.8,
        top_k: Some(50),
        ..Default::default()
    })?;
    
    let output_ids: Vec<u32> = output.iter().map(|f| f as u32).collect();
    println!("{}", tokenizer.decode(&output_ids));
    
    Ok(())
}
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GptModel                              │
├─────────────────────────────────────────────────────────────┤
│  wte: Embedding (tokens)     wpe: Embedding (positions)     │
│  blocks: Vec<GptBlock>       ln_f: LayerNorm                │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                        GptBlock                              │
├─────────────────────────────────────────────────────────────┤
│  ln_1 → CausalSelfAttention → residual                      │
│  ln_2 → GptMlp (fc→GELU→proj) → residual                    │
└─────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                   CausalSelfAttention                        │
├─────────────────────────────────────────────────────────────┤
│  1. QKV = c_attn(x)           [B, T, 3*C]                   │
│  2. Split Q, K, V             [B, T, C] each                │
│  3. Reshape to heads          [B, H, T, C/H]                │
│  4. scores = Q @ K.T / √d     [B, H, T, T]                  │
│  5. Causal mask (future=-∞)                                 │
│  6. attn = softmax(scores)                                  │
│  7. out = attn @ V → reshape → c_proj                       │
└─────────────────────────────────────────────────────────────┘
```

## 📊 GPT-2 Variants

| Variant | Embedding | Layers | Heads | Parameters |
|---------|-----------|--------|-------|------------|
| Small   | 768       | 12     | 12    | 124M       |
| Medium  | 1024      | 24     | 16    | 355M       |
| Large   | 1280      | 36     | 20    | 774M       |
| XL      | 1600      | 48     | 25    | 1.5B       |

## 🔧 Building

```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run the GPT-2 example
cargo run --example gpt2_generate -p rustml-nlp
```

## 📝 Tensor Operations

The `rustml-core` crate provides these key operations:

- **Creation**: `zeros`, `ones`, `randn`, `eye`, `tril`, `triu`, `arange`
- **Indexing**: `slice`, `select`, `cat`, `masked_fill`
- **Shape**: `reshape`, `unsqueeze`, `squeeze`, `transpose`, `permute`
- **Math**: `add`, `sub`, `mul`, `div`, `matmul`, `exp`, `log`, `sqrt`
- **Activations**: `relu`, `gelu`, `sigmoid`, `tanh`, `softmax`
- **Reductions**: `sum`, `mean`, `var`, `max`, `min`, `argmax`

## 📜 License

MIT OR Apache-2.0
