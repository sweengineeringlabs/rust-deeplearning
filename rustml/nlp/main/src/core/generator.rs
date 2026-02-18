//! High-level inference engine with streaming, beam search, and chat templates.
//!
//! Works with any `LanguageModel` implementation (GptModel, LlmModel, etc.)
//! and any `Tokenizer` implementation (BpeTokenizer, HFTokenizer, etc.).

use std::time::{Duration, Instant};

use crate::api::error::NlpResult;
use crate::api::types::LanguageModel;
use crate::core::sampling;
use rustml_tokenizer::Tokenizer;
use rayon::prelude::*;
use rustml_core::{DType, Tensor, f32_vec_to_bytes};
use rustml_nn::KVCache;

/// A segment of a chat template: either a special token (looked up by name)
/// or plain text (encoded normally by the tokenizer).
enum TemplateSegment {
    Special(String),
    Text(String),
}

/// Build template segments for a prompt given a chat template string.
fn build_template_segments(prompt: &str, template: &str) -> Vec<TemplateSegment> {
    if template.contains("<|user|>") || template.contains("<|system|>") {
        return vec![
            TemplateSegment::Special("<|system|>".into()),
            TemplateSegment::Text("\nYou are a helpful assistant.".into()),
            TemplateSegment::Special("</s>".into()),
            TemplateSegment::Text("\n".into()),
            TemplateSegment::Special("<|user|>".into()),
            TemplateSegment::Text(format!("\n{}", prompt)),
            TemplateSegment::Special("</s>".into()),
            TemplateSegment::Text("\n".into()),
            TemplateSegment::Special("<|assistant|>".into()),
            TemplateSegment::Text("\n".into()),
        ];
    }

    if template.contains("[INST]") {
        return vec![
            TemplateSegment::Special("[INST]".into()),
            TemplateSegment::Text(format!(" {} ", prompt)),
            TemplateSegment::Special("[/INST]".into()),
        ];
    }

    if template.contains("<|im_start|>") {
        return vec![
            TemplateSegment::Special("<|im_start|>".into()),
            TemplateSegment::Text("system\nYou are a helpful assistant.".into()),
            TemplateSegment::Special("<|im_end|>".into()),
            TemplateSegment::Text("\n".into()),
            TemplateSegment::Special("<|im_start|>".into()),
            TemplateSegment::Text(format!("user\n{}", prompt)),
            TemplateSegment::Special("<|im_end|>".into()),
            TemplateSegment::Text("\n".into()),
            TemplateSegment::Special("<|im_start|>".into()),
            TemplateSegment::Text("assistant\n".into()),
        ];
    }

    // Unknown template — encode the prompt as plain text
    eprintln!("[warn] unrecognized chat template format, falling back to plain text encoding");
    vec![TemplateSegment::Text(prompt.to_string())]
}

/// High-level generator combining a model, tokenizer, and sampling parameters.
///
/// Supports greedy/sampling generation, streaming output, beam search, and
/// chat template application.
pub struct Generator<'a> {
    model: &'a dyn LanguageModel,
    tokenizer: &'a dyn Tokenizer,
    temperature: f32,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    chat_template: Option<String>,
    deadline: Option<Instant>,
}

impl<'a> Generator<'a> {
    pub fn new(
        model: &'a dyn LanguageModel,
        tokenizer: &'a dyn Tokenizer,
        temperature: f32,
    ) -> Self {
        Self {
            model,
            tokenizer,
            temperature,
            eos_token_id: None,
            bos_token_id: None,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            chat_template: None,
            deadline: None,
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

    pub fn with_bos_token(mut self, bos: u32) -> Self {
        self.bos_token_id = Some(bos);
        self
    }

    pub fn with_chat_template(mut self, template: Option<String>) -> Self {
        self.chat_template = template;
        self
    }

    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_timeout(mut self, duration: Duration) -> Self {
        self.deadline = Some(Instant::now() + duration);
        self
    }

    /// Encode a prompt, applying the chat template if configured.
    fn encode_prompt(&self, prompt: &str) -> NlpResult<Vec<u32>> {
        let template = match self.chat_template {
            Some(ref tmpl) => tmpl,
            None => return Ok(self.tokenizer.encode(prompt)?),
        };

        let segments = build_template_segments(prompt, template);

        let mut token_ids = Vec::new();
        let mut all_resolved = true;

        for seg in &segments {
            match seg {
                TemplateSegment::Special(tok) => {
                    if let Some(id) = self.tokenizer.token_to_id(tok) {
                        token_ids.push(id);
                    } else {
                        all_resolved = false;
                        break;
                    }
                }
                TemplateSegment::Text(text) => {
                    if !text.is_empty() {
                        let ids = self.tokenizer.encode(text)?;
                        token_ids.extend(ids);
                    }
                }
            }
        }

        if all_resolved {
            return Ok(token_ids);
        }

        // Fallback: encode as a single formatted string
        let formatted: String = segments
            .iter()
            .map(|seg| match seg {
                TemplateSegment::Special(s) => s.as_str(),
                TemplateSegment::Text(t) => t.as_str(),
            })
            .collect();
        Ok(self.tokenizer.encode(&formatted)?)
    }

    /// Full sampling pipeline: rep_penalty → temperature → top_k → top_p → categorical/argmax
    fn sample_token(&self, logits: &[f32], past_tokens: &[u32]) -> u32 {
        if self.temperature < 1e-7 {
            return sampling::argmax(logits);
        }

        let mut logits = logits.to_vec();

        if let Some(penalty) = self.repetition_penalty {
            sampling::apply_repetition_penalty(&mut logits, past_tokens, penalty);
        }

        for v in logits.iter_mut() {
            *v /= self.temperature;
        }

        if let Some(k) = self.top_k {
            sampling::apply_top_k(&mut logits, k);
        }

        if let Some(p) = self.top_p {
            sampling::apply_top_p(&mut logits, p);
        }

        let mut rng = rand::thread_rng();
        sampling::sample_categorical(&logits, &mut rng)
    }

    fn make_cache(&self) -> KVCache {
        KVCache::new(
            self.model.num_layers(),
            self.model.max_sequence_length(),
            self.model.head_dim(),
            self.model.num_kv_heads(),
        )
    }

    fn prefill(&self, tokens: &[u32], cache: &mut KVCache) -> NlpResult<Vec<f32>> {
        let seq_len = tokens.len();
        let input_data: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        let input_bytes = f32_vec_to_bytes(input_data);
        let input = Tensor::new(input_bytes, vec![1, seq_len], DType::F32);

        let logits = self.model.forward_with_cache(&input, cache)?;
        cache.advance(seq_len);

        let logits_data: Vec<f32> = logits.iter().collect();
        let vocab_size = self.model.vocab_size();
        let start = (seq_len - 1) * vocab_size;
        let last_logits = &logits_data[start..start + vocab_size];

        Ok(last_logits.to_vec())
    }

    fn decode_step(&self, token: u32, cache: &mut KVCache) -> NlpResult<Vec<f32>> {
        let input_val = token as f32;
        let input_bytes = input_val.to_ne_bytes().to_vec();
        let input = Tensor::new(input_bytes, vec![1, 1], DType::F32);

        let logits = self.model.forward_with_cache(&input, cache)?;
        cache.advance(1);

        let logits_data: Vec<f32> = logits.iter().collect();
        let vocab_size = self.model.vocab_size();
        Ok(logits_data[0..vocab_size].to_vec())
    }

    /// Validate sampling parameters before generation.
    fn validate_params(&self) -> NlpResult<()> {
        if self.temperature < 0.0 {
            return Err(crate::api::error::NlpError::GenerationError(
                format!("temperature must be >= 0.0, got {}", self.temperature),
            ));
        }
        if let Some(k) = self.top_k {
            if k == 0 {
                return Err(crate::api::error::NlpError::GenerationError(
                    "top_k must be > 0".into(),
                ));
            }
        }
        if let Some(p) = self.top_p {
            if p <= 0.0 || p > 1.0 {
                return Err(crate::api::error::NlpError::GenerationError(
                    format!("top_p must be in (0.0, 1.0], got {}", p),
                ));
            }
        }
        if let Some(rp) = self.repetition_penalty {
            if rp <= 0.0 {
                return Err(crate::api::error::NlpError::GenerationError(
                    format!("repetition_penalty must be > 0.0, got {}", rp),
                ));
            }
        }
        Ok(())
    }

    /// Check if the generation deadline has been exceeded.
    fn check_deadline(&self) -> NlpResult<()> {
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return Err(crate::api::error::NlpError::GenerationError(
                    "generation deadline exceeded".into(),
                ));
            }
        }
        Ok(())
    }

    /// Generate text from a prompt. Returns the full output (prompt + generated).
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> NlpResult<String> {
        self.validate_params()?;
        let mut tokens = self.encode_prompt(prompt)?;
        if let Some(bos) = self.bos_token_id {
            tokens.insert(0, bos);
        }
        let mut cache = self.make_cache();

        let last_logits = self.prefill(&tokens, &mut cache)?;
        let mut next_token = self.sample_token(&last_logits, &tokens);

        if let Some(eos) = self.eos_token_id {
            if next_token == eos {
                return Ok(self.tokenizer.decode(&tokens)?);
            }
        }
        tokens.push(next_token);

        for _ in 1..max_tokens {
            self.check_deadline()?;
            let logits = self.decode_step(next_token, &mut cache)?;
            next_token = self.sample_token(&logits, &tokens);

            if let Some(eos) = self.eos_token_id {
                if next_token == eos {
                    break;
                }
            }
            tokens.push(next_token);
        }

        Ok(self.tokenizer.decode(&tokens)?)
    }

    /// Generate with a streaming callback. The callback receives each new token ID
    /// and returns `true` to continue or `false` to stop early.
    pub fn generate_stream<F: FnMut(u32) -> bool>(
        &self,
        prompt: &str,
        max_tokens: usize,
        mut callback: F,
    ) -> NlpResult<String> {
        self.validate_params()?;
        let mut tokens = self.encode_prompt(prompt)?;
        if let Some(bos) = self.bos_token_id {
            tokens.insert(0, bos);
        }
        let mut cache = self.make_cache();

        let last_logits = self.prefill(&tokens, &mut cache)?;
        let mut next_token = self.sample_token(&last_logits, &tokens);

        if let Some(eos) = self.eos_token_id {
            if next_token == eos {
                return Ok(self.tokenizer.decode(&tokens)?);
            }
        }
        tokens.push(next_token);
        if !callback(next_token) {
            return Ok(self.tokenizer.decode(&tokens)?);
        }

        for _ in 1..max_tokens {
            self.check_deadline()?;
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

        Ok(self.tokenizer.decode(&tokens)?)
    }

    /// Beam search generation. Maintains `beam_width` candidate sequences
    /// and returns the highest-scoring completed sequence.
    pub fn generate_beam(
        &self,
        prompt: &str,
        max_tokens: usize,
        beam_width: usize,
    ) -> NlpResult<String> {
        self.validate_params()?;
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        let mut cache = self.make_cache();

        let logits = self.prefill(&prompt_tokens, &mut cache)?;
        let log_probs = sampling::compute_log_probs(&logits);

        let top_tokens = sampling::top_n_indices(&log_probs, beam_width);
        let mut beams: Vec<BeamHypothesis> = Vec::with_capacity(beam_width);

        for &tok_idx in &top_tokens {
            let mut tokens = prompt_tokens.clone();
            tokens.push(tok_idx as u32);

            let beam_cache = cache.deep_clone()?;
            let finished = self
                .eos_token_id
                .is_some_and(|eos| tok_idx as u32 == eos);

            beams.push(BeamHypothesis {
                tokens,
                log_prob: log_probs[tok_idx],
                cache: beam_cache,
                finished,
            });
        }

        for _ in 1..max_tokens {
            self.check_deadline()?;
            if beams.iter().all(|b| b.finished) {
                break;
            }

            let mut candidates: Vec<BeamHypothesis> = Vec::new();

            for beam in beams.into_iter() {
                if beam.finished {
                    candidates.push(beam);
                    continue;
                }

                let last_token = *beam.tokens.last().ok_or_else(|| {
                    crate::api::error::NlpError::GenerationError(
                        "beam search: empty token sequence".into(),
                    )
                })?;
                let mut beam_cache = beam.cache;
                let logits = self.decode_step(last_token, &mut beam_cache)?;
                let lp = sampling::compute_log_probs(&logits);

                let top = sampling::top_n_indices(&lp, beam_width);
                for &tok_idx in &top {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(tok_idx as u32);

                    let new_cache = beam_cache.deep_clone()?;
                    let finished = self
                        .eos_token_id
                        .is_some_and(|eos| tok_idx as u32 == eos);

                    candidates.push(BeamHypothesis {
                        tokens: new_tokens,
                        log_prob: beam.log_prob + lp[tok_idx],
                        cache: new_cache,
                        finished,
                    });
                }
            }

            candidates.sort_unstable_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
            candidates.truncate(beam_width);
            beams = candidates;
        }

        beams.sort_unstable_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        let best = beams.first().ok_or_else(|| {
            crate::api::error::NlpError::GenerationError(
                "beam search: no candidate beams produced".into(),
            )
        })?;
        Ok(self.tokenizer.decode(&best.tokens)?)
    }

    /// Generate completions for multiple prompts sequentially.
    pub fn generate_batch(
        &self,
        prompts: &[&str],
        max_tokens: usize,
    ) -> NlpResult<Vec<String>> {
        prompts
            .iter()
            .map(|prompt| self.generate(prompt, max_tokens))
            .collect()
    }

    /// Generate completions for multiple prompts in parallel using rayon.
    ///
    /// Requires the model and tokenizer to be Sync (most implementations are).
    pub fn generate_batch_parallel(
        &self,
        prompts: &[&str],
        max_tokens: usize,
    ) -> NlpResult<Vec<String>> {
        prompts
            .par_iter()
            .map(|prompt| self.generate(prompt, max_tokens))
            .collect::<Result<Vec<_>, _>>()
    }
}

struct BeamHypothesis {
    tokens: Vec<u32>,
    log_prob: f32,
    cache: KVCache,
    finished: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::ModelConfig;
    use crate::core::model::LlmModel;
    use rustml_nn::PositionEncoding;

    /// A trivial tokenizer for testing: maps bytes to token IDs.
    struct ByteTokenizer;

    impl Tokenizer for ByteTokenizer {
        fn encode(&self, text: &str) -> rustml_tokenizer::TokenizerResult<Vec<u32>> {
            Ok(text.bytes().map(|b| b as u32).collect())
        }
        fn decode(&self, tokens: &[u32]) -> rustml_tokenizer::TokenizerResult<String> {
            let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
        fn vocab_size(&self) -> usize {
            256
        }
        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
    }

    fn tiny_model() -> LlmModel {
        let config = ModelConfig {
            dim: 64,
            hidden_dim: 256,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: None,
            vocab_size: 256,
            norm_eps: 1e-5,
            max_seq_len: 64,
            use_bias: Some(false),
            position_encoding: PositionEncoding::Learned,
            causal: true,
            rope_theta: 10000.0,
            bos_token_id: None,
            eos_token_id: None,
            chat_template: None,
            sliding_window: None,
            attn_logit_cap: None,
            embedding_scale: None,
            rms_norm_offset: None,
            attention_bias: None,
            parallel_residual: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            head_dim: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            rope_local_base_freq: None,
            rope_scaling_factor: None,
        };
        LlmModel::new(&config).unwrap()
    }

    #[test]
    fn test_generator_basic() {
        let model = tiny_model();
        let tokenizer = ByteTokenizer;
        let generator = Generator::new(&model, &tokenizer, 0.0); // greedy
        let result = generator.generate("AB", 4).unwrap();
        // Should produce something — at least the 2-char prompt
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_generator_stream() {
        let model = tiny_model();
        let tokenizer = ByteTokenizer;
        let generator = Generator::new(&model, &tokenizer, 0.7);
        let mut count = 0u32;
        let _result = generator
            .generate_stream("AB", 4, |_tok| {
                count += 1;
                true
            })
            .unwrap();
        assert!(count >= 1);
    }

    #[test]
    fn test_generator_beam() {
        let model = tiny_model();
        let tokenizer = ByteTokenizer;
        let generator = Generator::new(&model, &tokenizer, 0.0);
        let result = generator.generate_beam("AB", 4, 2).unwrap();
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_generator_batch() {
        let model = tiny_model();
        let tokenizer = ByteTokenizer;
        let generator = Generator::new(&model, &tokenizer, 0.0);
        let results = generator.generate_batch(&["A", "B"], 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_generator_with_sampling_params() {
        let model = tiny_model();
        let tokenizer = ByteTokenizer;
        let generator = Generator::new(&model, &tokenizer, 0.8)
            .with_top_k(10)
            .with_top_p(0.9)
            .with_repetition_penalty(1.1);
        let result = generator.generate("AB", 4).unwrap();
        assert!(result.len() >= 2);
    }
}
