//! Text Generation with various sampling strategies
//!
//! Supports:
//! - Greedy decoding
//! - Temperature sampling
//! - Top-k sampling
//! - Top-p (nucleus) sampling

use crate::api::error::NlpResult;
use crate::api::types::GenerationConfig;
use crate::core::gpt::GptModel;
use rand::Rng;
use rustml_core::Tensor;

/// Text generator using a GPT model
pub struct TextGenerator<'a> {
    model: &'a GptModel,
}

impl<'a> TextGenerator<'a> {
    /// Create a new text generator
    pub fn new(model: &'a GptModel) -> Self {
        Self { model }
    }

    /// Generate text autoregressively
    ///
    /// # Arguments
    /// * `input_ids` - Starting token IDs, shape [batch_size, seq_len] or [seq_len]
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token IDs including the input
    pub fn generate(&self, input_ids: &Tensor, config: &GenerationConfig) -> NlpResult<Tensor> {
        let mut rng = rand::thread_rng();

        // Ensure input is 2D [batch, seq]
        let is_1d = input_ids.ndim() == 1;
        let mut current_ids = if is_1d {
            input_ids.unsqueeze(0)?
        } else {
            input_ids.clone()
        };

        let max_length = self.model.max_sequence_length();

        for _ in 0..config.max_new_tokens {
            let seq_len = current_ids.shape()[1];

            // Check if we've exceeded max length
            if seq_len >= max_length {
                break;
            }

            // Truncate if needed (sliding window)
            let model_input = if seq_len > max_length {
                current_ids.slice(1, seq_len - max_length, seq_len)?
            } else {
                current_ids.clone()
            };

            // Forward pass
            let logits = self.model.forward(&model_input)?;

            // Get logits for the last position: [batch, vocab_size]
            let last_logits = logits.select(1, logits.shape()[1] - 1)?;

            // Apply repetition penalty
            let last_logits = if config.repetition_penalty != 1.0 {
                self.apply_repetition_penalty(&last_logits, &current_ids, config.repetition_penalty)?
            } else {
                last_logits
            };

            // Sample next token
            let next_token = if config.do_sample {
                self.sample(&last_logits, config, &mut rng)?
            } else {
                // Greedy: argmax
                last_logits.argmax(-1)?
            };

            // Check for EOS
            if let Some(eos_id) = config.eos_token_id {
                let next_token_id = next_token.get(&[0])? as u32;
                if next_token_id == eos_id {
                    break;
                }
            }

            // Append next token to sequence
            let next_token_2d = next_token.unsqueeze(-1)?;
            current_ids = Tensor::cat(&[&current_ids, &next_token_2d], 1)?;
        }

        // Return to original dimensionality if input was 1D
        if is_1d {
            Ok(current_ids.squeeze(0)?)
        } else {
            Ok(current_ids)
        }
    }

    /// Sample from logits using temperature and optional filtering
    fn sample<R: Rng>(
        &self,
        logits: &Tensor,
        config: &GenerationConfig,
        rng: &mut R,
    ) -> NlpResult<Tensor> {
        let batch_size = logits.shape()[0];
        let vocab_size = logits.shape()[1];

        let mut result = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            // Get logits for this batch element
            let batch_logits = logits.select(0, b)?;

            // Apply temperature
            let scaled_logits = if config.temperature != 1.0 {
                batch_logits.div_scalar(config.temperature)
            } else {
                batch_logits
            };

            // Convert to probabilities
            let mut probs = scaled_logits.softmax(-1)?;

            // Apply top-k filtering
            if let Some(k) = config.top_k {
                probs = self.top_k_filter(&probs, k)?;
            }

            // Apply top-p (nucleus) filtering
            if let Some(p) = config.top_p {
                probs = self.top_p_filter(&probs, p)?;
            }

            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                probs = probs.div_scalar(sum);
            }

            // Sample from distribution
            let token = self.sample_from_probs(&probs, rng)?;
            result.push(token as f32);
        }

        Ok(Tensor::from_vec(result, vec![batch_size])?)
    }

    /// Apply top-k filtering: keep only top k highest probability tokens
    fn top_k_filter(&self, probs: &Tensor, k: usize) -> NlpResult<Tensor> {
        let vocab_size = probs.shape()[0];
        let k = k.min(vocab_size);

        // Get indices and values sorted by probability
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zero out everything except top-k
        let mut filtered = vec![0.0f32; vocab_size];
        for (idx, val) in indexed.iter().take(k) {
            filtered[*idx] = *val;
        }

        Ok(Tensor::from_vec(filtered, vec![vocab_size])?)
    }

    /// Apply top-p (nucleus) filtering: keep tokens with cumulative prob <= p
    fn top_p_filter(&self, probs: &Tensor, p: f32) -> NlpResult<Tensor> {
        let vocab_size = probs.shape()[0];

        // Get sorted indices by probability (descending)
        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find cutoff where cumulative prob exceeds p
        let mut cumulative = 0.0;
        let mut cutoff_idx = vocab_size;

        for (i, (_, prob)) in indexed.iter().enumerate() {
            cumulative += prob;
            if cumulative > p {
                cutoff_idx = i + 1; // Include the token that pushed us over
                break;
            }
        }

        // Zero out everything after cutoff
        let mut filtered = vec![0.0f32; vocab_size];
        for (idx, val) in indexed.iter().take(cutoff_idx) {
            filtered[*idx] = *val;
        }

        Ok(Tensor::from_vec(filtered, vec![vocab_size])?)
    }

    /// Sample a token index from probability distribution
    fn sample_from_probs<R: Rng>(&self, probs: &Tensor, rng: &mut R) -> NlpResult<usize> {
        let r: f32 = rng.r#gen();
        let mut cumulative = 0.0;

        for (i, p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return Ok(i);
            }
        }

        // Fallback to last token (shouldn't happen with normalized probs)
        Ok(probs.numel() - 1)
    }

    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        generated_ids: &Tensor,
        penalty: f32,
    ) -> NlpResult<Tensor> {
        let batch_size = logits.shape()[0];
        let vocab_size = logits.shape()[1];

        let mut penalized_data = logits.to_vec();

        for b in 0..batch_size {
            // Get generated tokens for this batch
            let seq_len = generated_ids.shape()[1];
            for t in 0..seq_len {
                let token_id = generated_ids.get(&[b, t])? as usize;
                if token_id < vocab_size {
                    let idx = b * vocab_size + token_id;
                    let logit = penalized_data[idx];
                    // Penalize: divide positive logits, multiply negative logits
                    penalized_data[idx] = if logit > 0.0 {
                        logit / penalty
                    } else {
                        logit * penalty
                    };
                }
            }
        }

        Ok(Tensor::from_vec(penalized_data, logits.shape().to_vec())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::GptConfig;

    fn create_test_model() -> GptModel {
        let config = GptConfig {
            vocab_size: 100,
            n_positions: 32,
            n_embd: 64,
            n_layer: 2,
            n_head: 4,
            layer_norm_eps: 1e-5,
        };
        GptModel::new(config)
    }

    #[test]
    fn test_generation_config_presets() {
        let greedy = GenerationConfig::greedy(50);
        assert!(!greedy.do_sample);

        let temp = GenerationConfig::with_temperature(50, 0.8);
        assert!(temp.do_sample);
        assert_eq!(temp.temperature, 0.8);

        let topk = GenerationConfig::with_top_k(50, 40, 0.7);
        assert_eq!(topk.top_k, Some(40));

        let topp = GenerationConfig::with_top_p(50, 0.9, 0.7);
        assert_eq!(topp.top_p, Some(0.9));
    }

    #[test]
    fn test_generate_greedy() {
        let model = create_test_model();
        let generator = TextGenerator::new(&model);

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let config = GenerationConfig::greedy(5);

        let output = generator.generate(&input, &config).unwrap();
        // Output should have 3 + 5 = 8 tokens (or less if EOS)
        assert!(output.shape()[1] >= 3);
        assert!(output.shape()[1] <= 8);
    }

    #[test]
    fn test_top_k_filter() {
        let model = create_test_model();
        let generator = TextGenerator::new(&model);

        let probs = Tensor::from_vec(vec![0.1, 0.3, 0.2, 0.4], vec![4]).unwrap();
        let filtered = generator.top_k_filter(&probs, 2).unwrap();

        // Only top 2 (0.3 and 0.4) should be non-zero
        let filtered_vec: Vec<f32> = filtered.iter().collect();
        assert_eq!(filtered_vec[0], 0.0);
        assert!(filtered_vec[1] > 0.0); // 0.3
        assert_eq!(filtered_vec[2], 0.0);
        assert!(filtered_vec[3] > 0.0); // 0.4
    }

    #[test]
    fn test_top_p_filter() {
        let model = create_test_model();
        let generator = TextGenerator::new(&model);

        // Probs: 0.5, 0.3, 0.15, 0.05 (sorted desc: 0.5, 0.3, 0.15, 0.05)
        let probs = Tensor::from_vec(vec![0.5, 0.3, 0.15, 0.05], vec![4]).unwrap();
        let filtered = generator.top_p_filter(&probs, 0.85).unwrap();

        // Should keep 0.5 + 0.3 = 0.8, then add 0.15 to exceed 0.85
        let filtered_vec: Vec<f32> = filtered.iter().collect();
        assert!(filtered_vec[0] > 0.0); // 0.5
        assert!(filtered_vec[1] > 0.0); // 0.3
        assert!(filtered_vec[2] > 0.0); // 0.15 (cumsum 0.95 > 0.85, so include)
        assert_eq!(filtered_vec[3], 0.0); // 0.05 excluded
    }
}
