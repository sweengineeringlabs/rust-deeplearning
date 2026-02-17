use crate::attention::KVCache;
use crate::error::Result;
use super::generator::Generator;

struct BeamHypothesis {
    tokens: Vec<u32>,
    log_prob: f32,
    cache: KVCache,
    finished: bool,
}

/// Compute log-softmax of logits: log(softmax(x))
fn compute_log_probs(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = logits.iter()
        .map(|&v| (v - max_val).exp())
        .sum::<f32>()
        .ln()
        + max_val;
    logits.iter().map(|&v| v - log_sum_exp).collect()
}

/// Return indices of the top-n values in descending order.
fn top_n_indices(values: &[f32], n: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    indexed.iter().take(n).map(|&(i, _)| i).collect()
}

impl<'a> Generator<'a> {
    /// Beam search generation. Maintains `beam_width` candidate sequences
    /// and returns the highest-scoring completed sequence.
    pub fn generate_beam(
        &self,
        prompt: &str,
        max_tokens: usize,
        beam_width: usize,
    ) -> Result<String> {
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        let mut cache = self.make_cache();

        // Prefill with prompt
        let logits = self.prefill(&prompt_tokens, &mut cache)?;
        let log_probs = compute_log_probs(&logits);

        // Initialize beams from top-k tokens of first prediction
        let top_tokens = top_n_indices(&log_probs, beam_width);
        let mut beams: Vec<BeamHypothesis> = Vec::with_capacity(beam_width);

        for &tok_idx in &top_tokens {
            let mut tokens = prompt_tokens.clone();
            tokens.push(tok_idx as u32);

            let beam_cache = cache.deep_clone()?;

            let finished = self.eos_token_id.map_or(false, |eos| tok_idx as u32 == eos);

            beams.push(BeamHypothesis {
                tokens,
                log_prob: log_probs[tok_idx],
                cache: beam_cache,
                finished,
            });
        }

        // Decode loop
        for _ in 1..max_tokens {
            // Check if all beams are finished
            if beams.iter().all(|b| b.finished) {
                break;
            }

            let mut candidates: Vec<BeamHypothesis> = Vec::new();

            // Keep finished beams as-is
            for beam in beams.into_iter() {
                if beam.finished {
                    candidates.push(beam);
                    continue;
                }

                let last_token = *beam.tokens.last().unwrap();
                let mut beam_cache = beam.cache;
                let logits = self.decode_step(last_token, &mut beam_cache)?;
                let lp = compute_log_probs(&logits);

                // Expand this beam by top beam_width tokens
                let top = top_n_indices(&lp, beam_width);
                for &tok_idx in &top {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(tok_idx as u32);

                    let new_cache = beam_cache.deep_clone()?;
                    let finished = self.eos_token_id.map_or(false, |eos| tok_idx as u32 == eos);

                    candidates.push(BeamHypothesis {
                        tokens: new_tokens,
                        log_prob: beam.log_prob + lp[tok_idx],
                        cache: new_cache,
                        finished,
                    });
                }
            }

            // Prune to top beam_width by log_prob
            candidates.sort_unstable_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
            candidates.truncate(beam_width);
            beams = candidates;
        }

        // Return best beam
        beams.sort_unstable_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
        let best = &beams[0];
        self.tokenizer.decode(&best.tokens)
    }
}
