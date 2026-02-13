use crate::core::tensor::{Tensor, DType};
use crate::error::Result;
use crate::models::LlmModel;
use crate::tokenization::Tokenizer;
use crate::attention::KVCache;

use super::sampling;

pub struct Generator<'a> {
    pub(crate) model: &'a LlmModel,
    pub(crate) tokenizer: &'a dyn Tokenizer,
    pub(crate) temperature: f32,
    pub eos_token_id: Option<u32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

impl<'a> Generator<'a> {
    pub fn new(model: &'a LlmModel, tokenizer: &'a dyn Tokenizer, temperature: f32) -> Self {
        Self {
            model,
            tokenizer,
            temperature,
            eos_token_id: None,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
        }
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }

    pub fn with_eos_token(mut self, eos: u32) -> Self {
        self.eos_token_id = Some(eos);
        self
    }

    /// Full sampling pipeline: rep_penalty → temperature → top_k → top_p → categorical/argmax
    fn sample_token(&self, logits: &[f32], past_tokens: &[u32]) -> u32 {
        if self.temperature < 1e-7 {
            return sampling::argmax(logits);
        }

        let mut logits = logits.to_vec();

        // 1. Repetition penalty
        if let Some(penalty) = self.repetition_penalty {
            sampling::apply_repetition_penalty(&mut logits, past_tokens, penalty);
        }

        // 2. Temperature scaling
        for v in logits.iter_mut() {
            *v /= self.temperature;
        }

        // 3. Top-k
        if let Some(k) = self.top_k {
            sampling::apply_top_k(&mut logits, k);
        }

        // 4. Top-p
        if let Some(p) = self.top_p {
            sampling::apply_top_p(&mut logits, p);
        }

        // 5. Categorical sample
        let mut rng = rand::thread_rng();
        sampling::sample_categorical(&logits, &mut rng)
    }

    pub(crate) fn make_cache(&self) -> KVCache {
        let head_dim = self.model.d_model / self.model.n_heads;
        KVCache::new(
            self.model.n_layers,
            self.model.max_seq_len,
            head_dim,
            self.model.n_kv_heads,
        )
    }

    pub(crate) fn prefill(&self, tokens: &[u32], cache: &mut KVCache) -> Result<Vec<f32>> {
        let seq_len = tokens.len();
        let mut input_data = Vec::with_capacity(seq_len);
        for &t in tokens {
            input_data.push(t as f32);
        }

        let input_bytes = crate::core::tensor::f32_vec_to_bytes(input_data);
        let input = Tensor::new(input_bytes, vec![1, seq_len], DType::F32);

        let logits = self.model.forward_with_cache(&input, cache)?;
        cache.advance(seq_len);

        let logits_data = logits.as_slice_f32()?;
        let vocab_size = self.model.vocab_size;
        let start = (seq_len - 1) * vocab_size;
        Ok(logits_data[start..start + vocab_size].to_vec())
    }

    pub(crate) fn decode_step(&self, token: u32, cache: &mut KVCache) -> Result<Vec<f32>> {
        let input_val = token as f32;
        let input_bytes = input_val.to_ne_bytes().to_vec();
        let input = Tensor::new(input_bytes, vec![1, 1], DType::F32);

        let logits = self.model.forward_with_cache(&input, cache)?;
        cache.advance(1);

        let logits_data = logits.as_slice_f32()?;
        let vocab_size = self.model.vocab_size;
        Ok(logits_data[0..vocab_size].to_vec())
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let mut tokens = self.tokenizer.encode(prompt)?;
        let mut cache = self.make_cache();

        // Prefill
        let last_logits = self.prefill(&tokens, &mut cache)?;
        let mut next_token = self.sample_token(&last_logits, &tokens);

        if let Some(eos) = self.eos_token_id {
            if next_token == eos {
                return self.tokenizer.decode(&tokens);
            }
        }
        tokens.push(next_token);

        // Decode loop
        for _ in 0..max_tokens {
            let logits = self.decode_step(next_token, &mut cache)?;
            next_token = self.sample_token(&logits, &tokens);

            if let Some(eos) = self.eos_token_id {
                if next_token == eos {
                    break;
                }
            }
            tokens.push(next_token);
        }

        self.tokenizer.decode(&tokens)
    }

    /// Generate tokens with a streaming callback. The callback receives each
    /// new token ID and returns `true` to continue or `false` to stop early.
    pub fn generate_stream<F: FnMut(u32) -> bool>(
        &self,
        prompt: &str,
        max_tokens: usize,
        mut callback: F,
    ) -> Result<String> {
        let mut tokens = self.tokenizer.encode(prompt)?;
        let mut cache = self.make_cache();

        // Prefill
        let last_logits = self.prefill(&tokens, &mut cache)?;
        let mut next_token = self.sample_token(&last_logits, &tokens);

        if let Some(eos) = self.eos_token_id {
            if next_token == eos {
                return self.tokenizer.decode(&tokens);
            }
        }
        tokens.push(next_token);
        if !callback(next_token) {
            return self.tokenizer.decode(&tokens);
        }

        // Decode loop
        for _ in 0..max_tokens {
            let logits = self.decode_step(next_token, &mut cache)?;
            next_token = self.sample_token(&logits, &tokens);

            if let Some(eos) = self.eos_token_id {
                if next_token == eos {
                    break;
                }
            }
            tokens.push(next_token);
            if !callback(next_token) {
                break;
            }
        }

        self.tokenizer.decode(&tokens)
    }
}
