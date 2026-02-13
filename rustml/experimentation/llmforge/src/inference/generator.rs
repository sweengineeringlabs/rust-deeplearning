use crate::core::tensor::{Tensor, DType};
use crate::error::Result;
use crate::models::LlmModel;
use crate::tokenization::Tokenizer;
use crate::attention::KVCache;

pub struct Generator<'a> {
    model: &'a LlmModel,
    tokenizer: &'a dyn Tokenizer,
    temperature: f32,
    pub eos_token_id: Option<u32>,
}

impl<'a> Generator<'a> {
    pub fn new(model: &'a LlmModel, tokenizer: &'a dyn Tokenizer, temperature: f32) -> Self {
        Self {
            model,
            tokenizer,
            temperature,
            eos_token_id: None,
        }
    }

    fn sample_token(&self, logits: &[f32]) -> u32 {
        if self.temperature < 1e-7 {
            // Greedy: argmax
            let mut max_val = f32::NEG_INFINITY;
            let mut best = 0u32;
            for (i, &val) in logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    best = i as u32;
                }
            }
            best
        } else {
            // Temperature-scaled softmax sampling
            let scaled: Vec<f32> = logits.iter().map(|&v| v / self.temperature).collect();
            let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let probs: Vec<f32> = exp.iter().map(|&v| v / sum).collect();

            let mut rng = rand::thread_rng();
            let r: f32 = rand::Rng::gen(&mut rng);
            let mut cumsum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    return i as u32;
                }
            }
            (probs.len() - 1) as u32
        }
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let mut tokens = self.tokenizer.encode(prompt)?;

        let head_dim = self.model.d_model / self.model.n_heads;
        let mut cache = KVCache::new(
            self.model.n_layers,
            self.model.max_seq_len,
            head_dim,
            self.model.n_heads
        );

        // 1. Prefill
        let seq_len = tokens.len();
        let mut input_data = Vec::with_capacity(seq_len);
        for &t in &tokens {
            input_data.push(t as f32);
        }

        let input_bytes = crate::core::tensor::f32_vec_to_bytes(input_data);
        let input = Tensor::new(input_bytes, vec![1, seq_len], DType::F32);

        let logits = self.model.forward_with_cache(&input, &mut cache)?;
        cache.advance(seq_len);

        // Get last token logits
        let logits_data = logits.as_slice_f32()?;
        let vocab_size = self.model.vocab_size;
        let start = (seq_len - 1) * vocab_size;
        let last_logits = &logits_data[start..start + vocab_size];

        let mut next_token = self.sample_token(last_logits);

        // Check EOS
        if let Some(eos) = self.eos_token_id {
            if next_token == eos {
                return self.tokenizer.decode(&tokens);
            }
        }
        tokens.push(next_token);

        // 2. Decode Loop
        for _ in 0..max_tokens {
            // Prepare input [1, 1]
            let input_val = next_token as f32;
            let input_bytes = input_val.to_ne_bytes().to_vec();
            let input = Tensor::new(input_bytes, vec![1, 1], DType::F32);

            let logits = self.model.forward_with_cache(&input, &mut cache)?;
            cache.advance(1);

            // Get logits (shape [1, 1, Vocab])
            let logits_data = logits.as_slice_f32()?;
            let last_logits = &logits_data[0..vocab_size];

            next_token = self.sample_token(last_logits);

            // Check EOS
            if let Some(eos) = self.eos_token_id {
                if next_token == eos {
                    break;
                }
            }
            tokens.push(next_token);
        }

        self.tokenizer.decode(&tokens)
    }
}
