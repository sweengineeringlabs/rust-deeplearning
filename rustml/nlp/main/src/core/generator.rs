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

/// Build template segments for a multi-turn conversation given a chat template string.
/// Messages are `(role, content)` pairs where role is `"user"` or `"assistant"`.
fn build_multi_turn_segments(messages: &[(&str, &str)], template: &str) -> Vec<TemplateSegment> {
    let mut segments = Vec::new();

    // Gemma format: <start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n
    if template.contains("<start_of_turn>") {
        for &(role, content) in messages {
            let model_role = if role == "assistant" { "model" } else { role };
            segments.push(TemplateSegment::Special("<start_of_turn>".into()));
            segments.push(TemplateSegment::Text(format!("{}\n{}", model_role, content)));
            segments.push(TemplateSegment::Special("<end_of_turn>".into()));
            segments.push(TemplateSegment::Text("\n".into()));
        }
        // Start the assistant turn
        segments.push(TemplateSegment::Special("<start_of_turn>".into()));
        segments.push(TemplateSegment::Text("model\n".into()));
        return segments;
    }

    // ChatML format: <|system|>\n...\n<|user|>\n{msg}</s>\n<|assistant|>\n{resp}</s>\n...
    if template.contains("<|user|>") || template.contains("<|system|>") {
        segments.push(TemplateSegment::Special("<|system|>".into()));
        segments.push(TemplateSegment::Text("\nYou are a helpful assistant.".into()));
        segments.push(TemplateSegment::Special("</s>".into()));
        segments.push(TemplateSegment::Text("\n".into()));

        for &(role, content) in messages {
            let tag = if role == "user" { "<|user|>" } else { "<|assistant|>" };
            segments.push(TemplateSegment::Special(tag.into()));
            segments.push(TemplateSegment::Text(format!("\n{}", content)));
            segments.push(TemplateSegment::Special("</s>".into()));
            segments.push(TemplateSegment::Text("\n".into()));
        }
        segments.push(TemplateSegment::Special("<|assistant|>".into()));
        segments.push(TemplateSegment::Text("\n".into()));
        return segments;
    }

    // Llama/Mistral format: [INST] msg [/INST] resp [INST] msg2 [/INST]
    if template.contains("[INST]") {
        for &(role, content) in messages {
            if role == "user" {
                segments.push(TemplateSegment::Special("[INST]".into()));
                segments.push(TemplateSegment::Text(format!(" {} ", content)));
                segments.push(TemplateSegment::Special("[/INST]".into()));
            } else {
                segments.push(TemplateSegment::Text(format!(" {} ", content)));
            }
        }
        return segments;
    }

    // Qwen format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n...
    if template.contains("<|im_start|>") {
        segments.push(TemplateSegment::Special("<|im_start|>".into()));
        segments.push(TemplateSegment::Text("system\nYou are a helpful assistant.".into()));
        segments.push(TemplateSegment::Special("<|im_end|>".into()));
        segments.push(TemplateSegment::Text("\n".into()));

        for &(role, content) in messages {
            segments.push(TemplateSegment::Special("<|im_start|>".into()));
            segments.push(TemplateSegment::Text(format!("{}\n{}", role, content)));
            segments.push(TemplateSegment::Special("<|im_end|>".into()));
            segments.push(TemplateSegment::Text("\n".into()));
        }
        segments.push(TemplateSegment::Special("<|im_start|>".into()));
        segments.push(TemplateSegment::Text("assistant\n".into()));
        return segments;
    }

    // Unknown template — plain text fallback
    eprintln!("[warn] unrecognized chat template format for multi-turn, falling back to plain text");
    let text: String = messages
        .iter()
        .map(|(role, content)| format!("{}: {}\n", role, content))
        .collect();
    segments.push(TemplateSegment::Text(text));
    segments.push(TemplateSegment::Text("assistant: ".into()));
    segments
}

/// High-level generator combining a model, tokenizer, and sampling parameters.
///
/// Supports greedy/sampling generation, streaming output, beam search, and
/// chat template application.
pub struct Generator<'a> {
    model: &'a (dyn LanguageModel + Sync),
    tokenizer: &'a (dyn Tokenizer + Sync),
    temperature: f32,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    chat_template: Option<String>,
    deadline: Option<Instant>,
    context_len_override: Option<usize>,
    /// Use pre-allocated SamplingBuffer for top-p (default true).
    /// When false, allocates fresh Vec each token (baseline path).
    use_buffered_sampling: bool,
}

impl<'a> Generator<'a> {
    pub fn new(
        model: &'a (dyn LanguageModel + Sync),
        tokenizer: &'a (dyn Tokenizer + Sync),
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
            context_len_override: None,
            use_buffered_sampling: true,
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

    pub fn with_context_len(mut self, context_len: usize) -> Self {
        self.context_len_override = Some(context_len);
        self
    }

    /// Toggle buffered sampling. When true (default), reuses pre-allocated
    /// buffers for logits and top-p sorting. When false, allocates fresh
    /// each token (baseline path for A/B benchmarking).
    pub fn with_buffered_sampling(mut self, enabled: bool) -> Self {
        self.use_buffered_sampling = enabled;
        self
    }

    /// Apply an optimization profile to this generator's sampling behavior.
    pub fn with_optimization_profile(self, profile: rustml_core::OptProfile) -> Self {
        self.with_buffered_sampling(profile.use_buffered_sampling())
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

    /// Full sampling pipeline: rep_penalty → temperature → top_k → top_p → categorical/argmax.
    ///
    /// When `use_buffered_sampling` is true (default), reuses the pre-allocated
    /// `SamplingBuffer` to avoid per-token heap allocations.
    /// When false, allocates fresh each call (baseline path).
    fn sample_token(&self, logits: &[f32], past_tokens: &[u32], buf: &mut sampling::SamplingBuffer) -> u32 {
        if self.temperature < 1e-7 {
            return sampling::argmax(logits);
        }

        if self.use_buffered_sampling {
            // Reuse pre-allocated logits buffer instead of allocating a new Vec each token
            buf.logits.clear();
            buf.logits.extend_from_slice(logits);

            if let Some(penalty) = self.repetition_penalty {
                sampling::apply_repetition_penalty(&mut buf.logits, past_tokens, penalty);
            }

            for v in buf.logits.iter_mut() {
                *v /= self.temperature;
            }

            if let Some(k) = self.top_k {
                sampling::apply_top_k(&mut buf.logits, k);
            }

            if let Some(p) = self.top_p {
                sampling::apply_top_p_buffered(&mut buf.logits, p, &mut buf.sort_buf);
            }

            let mut rng = rand::thread_rng();
            sampling::sample_categorical(&buf.logits, &mut rng)
        } else {
            // Allocating baseline path
            let mut logits_copy = logits.to_vec();

            if let Some(penalty) = self.repetition_penalty {
                sampling::apply_repetition_penalty(&mut logits_copy, past_tokens, penalty);
            }

            for v in logits_copy.iter_mut() {
                *v /= self.temperature;
            }

            if let Some(k) = self.top_k {
                sampling::apply_top_k(&mut logits_copy, k);
            }

            if let Some(p) = self.top_p {
                sampling::apply_top_p(&mut logits_copy, p);
            }

            let mut rng = rand::thread_rng();
            sampling::sample_categorical(&logits_copy, &mut rng)
        }
    }

    fn make_cache(&self) -> KVCache {
        let max_seq = self.model.max_sequence_length();
        let effective = match self.context_len_override {
            Some(n) => n.min(max_seq),
            None => max_seq,
        };
        KVCache::new(
            self.model.num_layers(),
            effective,
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

        let logits_data = logits.to_vec();
        let vocab_size = self.model.vocab_size();
        let start = (seq_len - 1) * vocab_size;
        Ok(logits_data[start..start + vocab_size].to_vec())
    }

    fn decode_step(&self, token: u32, cache: &mut KVCache) -> NlpResult<Vec<f32>> {
        let input_val = token as f32;
        let input_bytes = input_val.to_ne_bytes().to_vec();
        let input = Tensor::new(input_bytes, vec![1, 1], DType::F32);

        let logits = self.model.forward_with_cache(&input, cache)?;
        cache.advance(1);

        Ok(logits.to_vec())
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
        let mut sampling_buf = sampling::SamplingBuffer::new(self.model.vocab_size());

        let _t_prefill = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
        let last_logits = self.prefill(&tokens, &mut cache)?;
        if let Some(t) = _t_prefill {
            log::debug!("[perf] generator::prefill tokens={} {:.3}ms", tokens.len(), t.elapsed().as_secs_f64() * 1000.0);
        }

        let _t_sample = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
        let mut next_token = self.sample_token(&last_logits, &tokens, &mut sampling_buf);
        if let Some(t) = _t_sample {
            log::debug!("[perf] generator::sample {:.3}ms", t.elapsed().as_secs_f64() * 1000.0);
        }

        if let Some(eos) = self.eos_token_id {
            if next_token == eos {
                return Ok(self.tokenizer.decode(&tokens)?);
            }
        }
        tokens.push(next_token);

        for _ in 1..max_tokens {
            self.check_deadline()?;
            let _t_step = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };

            let _t_model = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
            let logits = self.decode_step(next_token, &mut cache)?;
            let model_ms = _t_model.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            let _t_s = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
            next_token = self.sample_token(&logits, &tokens, &mut sampling_buf);
            let sample_ms = _t_s.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            if let Some(t) = _t_step {
                log::debug!("[perf] generator::decode_step token={} model={:.3}ms sample={:.3}ms total={:.3}ms",
                    tokens.len(), model_ms, sample_ms, t.elapsed().as_secs_f64() * 1000.0);
            }

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
        let mut sampling_buf = sampling::SamplingBuffer::new(self.model.vocab_size());

        let _t_prefill = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
        let last_logits = self.prefill(&tokens, &mut cache)?;
        if let Some(t) = _t_prefill {
            log::debug!("[perf] generator::prefill tokens={} {:.3}ms", tokens.len(), t.elapsed().as_secs_f64() * 1000.0);
        }

        let _t_sample = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
        let mut next_token = self.sample_token(&last_logits, &tokens, &mut sampling_buf);
        if let Some(t) = _t_sample {
            log::debug!("[perf] generator::sample {:.3}ms", t.elapsed().as_secs_f64() * 1000.0);
        }

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
            let _t_step = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };

            let _t_model = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
            let logits = self.decode_step(next_token, &mut cache)?;
            let model_ms = _t_model.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            let _t_s = if log::log_enabled!(log::Level::Debug) { Some(Instant::now()) } else { None };
            next_token = self.sample_token(&logits, &tokens, &mut sampling_buf);
            let sample_ms = _t_s.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            if let Some(t) = _t_step {
                log::debug!("[perf] generator::decode_step token={} model={:.3}ms sample={:.3}ms total={:.3}ms",
                    tokens.len(), model_ms, sample_ms, t.elapsed().as_secs_f64() * 1000.0);
            }

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

    /// Encode a multi-turn conversation, applying the chat template.
    /// Messages are `(role, content)` pairs where role is `"user"` or `"assistant"`.
    pub fn encode_conversation(&self, messages: &[(&str, &str)]) -> NlpResult<Vec<u32>> {
        let template = match self.chat_template {
            Some(ref tmpl) => tmpl,
            None => {
                // No chat template — format as plain text
                let text: String = messages
                    .iter()
                    .map(|(role, content)| format!("{}: {}\n", role, content))
                    .collect();
                let text = format!("{}assistant: ", text);
                return Ok(self.tokenizer.encode(&text)?);
            }
        };

        let segments = build_multi_turn_segments(messages, template);

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

    /// Resolve template-specific end-of-turn token IDs.
    fn resolve_stop_tokens(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        let candidates = ["<end_of_turn>", "<|im_end|>", "</s>", "<|eot_id|>"];
        for name in &candidates {
            if let Some(id) = self.tokenizer.token_to_id(name) {
                ids.push(id);
            }
        }
        ids
    }

    /// Check if a token is a stop token (EOS or end-of-turn).
    fn is_stop_token(&self, token: u32, end_of_turn_ids: &[u32]) -> bool {
        if let Some(eos) = self.eos_token_id {
            if token == eos {
                return true;
            }
        }
        end_of_turn_ids.contains(&token)
    }

    /// Generate a single assistant turn in a multi-turn conversation with streaming.
    ///
    /// Encodes the full conversation history, runs prefill + decode with the
    /// streaming callback, and returns only the assistant's new response text.
    /// Stops on EOS or template-specific end-of-turn tokens.
    pub fn generate_turn_stream<F: FnMut(u32) -> bool>(
        &self,
        messages: &[(&str, &str)],
        max_tokens: usize,
        mut callback: F,
    ) -> NlpResult<String> {
        self.validate_params()?;
        let mut tokens = self.encode_conversation(messages)?;
        if let Some(bos) = self.bos_token_id {
            tokens.insert(0, bos);
        }
        let prompt_len = tokens.len();
        let mut cache = self.make_cache();
        let mut sampling_buf = sampling::SamplingBuffer::new(self.model.vocab_size());

        let end_of_turn_ids = self.resolve_stop_tokens();

        let last_logits = self.prefill(&tokens, &mut cache)?;
        let mut next_token = self.sample_token(&last_logits, &tokens, &mut sampling_buf);

        if self.is_stop_token(next_token, &end_of_turn_ids) {
            return Ok(String::new());
        }
        tokens.push(next_token);
        if !callback(next_token) {
            let response_tokens = &tokens[prompt_len..];
            return Ok(self.tokenizer.decode(response_tokens)?);
        }

        for _ in 1..max_tokens {
            self.check_deadline()?;
            let logits = self.decode_step(next_token, &mut cache)?;
            next_token = self.sample_token(&logits, &tokens, &mut sampling_buf);

            if self.is_stop_token(next_token, &end_of_turn_ids) {
                break;
            }
            tokens.push(next_token);
            if !callback(next_token) {
                break;
            }
        }

        let response_tokens = &tokens[prompt_len..];
        Ok(self.tokenizer.decode(response_tokens)?)
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

    // ==================== Configurable buffered sampling tests ====================

    #[test]
    fn test_generator_buffered_vs_unbuffered_greedy() {
        // Greedy (temperature=0) should produce identical output regardless of
        // use_buffered_sampling — argmax doesn't go through the buffer path.
        let model = tiny_model();
        let tokenizer = ByteTokenizer;

        let gen_buf = Generator::new(&model, &tokenizer, 0.0)
            .with_buffered_sampling(true);
        let gen_alloc = Generator::new(&model, &tokenizer, 0.0)
            .with_buffered_sampling(false);

        let out1 = gen_buf.generate("AB", 8).unwrap();
        let out2 = gen_alloc.generate("AB", 8).unwrap();
        assert_eq!(out1, out2, "greedy output differs between buffered and unbuffered");
    }

    #[test]
    fn test_generator_unbuffered_basic() {
        // Ensure the unbuffered path doesn't crash
        let model = tiny_model();
        let tokenizer = ByteTokenizer;
        let generator = Generator::new(&model, &tokenizer, 0.7)
            .with_buffered_sampling(false)
            .with_top_p(0.9);
        let result = generator.generate("AB", 4).unwrap();
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_generator_with_optimization_profile() {
        use rustml_core::OptProfile;
        let model = tiny_model();
        let tokenizer = ByteTokenizer;

        // Baseline profile disables buffered sampling
        let generator_base = Generator::new(&model, &tokenizer, 0.0)
            .with_optimization_profile(OptProfile::Baseline);
        let out = generator_base.generate("AB", 4).unwrap();
        assert!(out.len() >= 2);

        // Optimized profile enables buffered sampling
        let generator_opt = Generator::new(&model, &tokenizer, 0.0)
            .with_optimization_profile(OptProfile::Optimized);
        let out2 = generator_opt.generate("AB", 4).unwrap();
        assert_eq!(out, out2, "optimized vs baseline greedy output should match");
    }
}
