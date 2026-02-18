//! Sampling functions for language model inference.
//!
//! Operates on raw logit slices (not Tensor) for efficiency.

use rand::Rng;

/// Return the index of the maximum value in the logit slice.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            best = i as u32;
        }
    }
    best
}

/// Apply repetition penalty (HuggingFace convention):
/// - positive logits are divided by `penalty`
/// - negative logits are multiplied by `penalty`
pub fn apply_repetition_penalty(logits: &mut [f32], past_tokens: &[u32], penalty: f32) {
    for &tok in past_tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Keep only the top-k logits; set the rest to NEG_INFINITY.
/// If k >= logits.len(), this is a no-op.
///
/// For small k (<=1024), uses a compact buffer instead of copying the full
/// vocab, reducing allocation from ~1 MB to ~8 KB for typical vocab sizes.
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k >= logits.len() {
        return;
    }

    let threshold = if k <= 1024 {
        // Small k: collect top-k candidates in a compact buffer (≤2k elements)
        // instead of cloning the entire vocab-sized slice.
        let mut buf: Vec<f32> = Vec::with_capacity(2 * k);
        for &v in logits.iter() {
            buf.push(v);
            if buf.len() == 2 * k {
                let pivot = buf.len() - k;
                buf.select_nth_unstable_by(pivot, |a, b| a.total_cmp(b));
                buf.drain(..pivot);
            }
        }
        if buf.len() > k {
            let pivot = buf.len() - k;
            buf.select_nth_unstable_by(pivot, |a, b| a.total_cmp(b));
            buf[pivot]
        } else {
            *buf.iter().min_by(|a, b| a.total_cmp(b)).unwrap()
        }
    } else {
        // Large k: full copy + select_nth (original approach)
        let mut vals: Vec<f32> = logits.to_vec();
        let pivot = vals.len() - k;
        vals.select_nth_unstable_by(pivot, |a: &f32, b: &f32| a.total_cmp(b));
        vals[pivot]
    };

    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Apply nucleus (top-p) sampling: keep the smallest set of tokens whose
/// cumulative probability exceeds `p`, mask the rest to NEG_INFINITY.
///
/// Uses a single `Vec<(f32, usize)>` allocation instead of 4 separate Vecs.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }

    // Stable softmax — compute sum without allocating
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();

    // Single allocation: (probability, original_index) pairs
    let mut prob_idx: Vec<(f32, usize)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| ((v - max_val).exp() / sum, i))
        .collect();

    // Sort by probability descending
    prob_idx.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    // Find cutoff index where cumulative probability reaches p
    let mut cumsum = 0.0f32;
    let mut cutoff = prob_idx.len();
    for (i, &(prob, _)) in prob_idx.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff = i + 1;
            break;
        }
    }

    // Mask tokens beyond the cutoff directly
    for &(_, idx) in &prob_idx[cutoff..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Sample from a categorical distribution defined by logits (not probabilities).
/// Applies stable softmax internally then draws using the given RNG.
///
/// Zero-allocation: uses two streaming passes instead of materializing
/// `exps` and `probs` Vecs (saves ~2 MB per call for 256K vocab).
pub fn sample_categorical<R: Rng>(logits: &[f32], rng: &mut R) -> u32 {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // First pass: compute sum(exp(logit - max))
    let sum: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();

    // Second pass: cumulative sampling against r * sum threshold
    let threshold = rng.r#gen::<f32>() * sum;
    let mut cumsum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        cumsum += (v - max_val).exp();
        if cumsum > threshold {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

/// Compute log-softmax of logits: log(softmax(x))
pub fn compute_log_probs(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = logits
        .iter()
        .map(|&v| (v - max_val).exp())
        .sum::<f32>()
        .ln()
        + max_val;
    logits.iter().map(|&v| v - log_sum_exp).collect()
}

/// Return indices of the top-n values in descending order.
pub fn top_n_indices(values: &[f32], n: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    indexed.iter().take(n).map(|&(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn test_top_k_basic() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count >= 2);
        assert!(logits[1].is_finite()); // 5.0
        assert!(logits[4].is_finite()); // 4.0
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_top_k(&mut logits, 10);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_p_keeps_minimum_set() {
        let mut logits = vec![10.0, -10.0, -10.0, -10.0];
        apply_top_p(&mut logits, 0.9);
        assert!(logits[0].is_finite());
        for &v in &logits[1..] {
            assert_eq!(v, f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_top_p_one_keeps_all() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let original = logits.clone();
        apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repetition_penalty_positive_logits() {
        let mut logits = vec![2.0, 4.0, 6.0];
        apply_repetition_penalty(&mut logits, &[0, 2], 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 4.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut logits = vec![-2.0, 1.0, -3.0];
        apply_repetition_penalty(&mut logits, &[0, 2], 2.0);
        assert!((logits[0] - (-4.0)).abs() < 1e-6);
        assert!((logits[1] - 1.0).abs() < 1e-6);
        assert!((logits[2] - (-6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sample_categorical_deterministic() {
        let logits = vec![f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            assert_eq!(sample_categorical(&logits, &mut rng), 1);
        }
    }

    #[test]
    fn test_sample_categorical_distribution() {
        let logits = vec![0.0; 4];
        let mut rng = rand::thread_rng();
        let mut counts = [0u32; 4];
        let n = 4000;
        for _ in 0..n {
            let idx = sample_categorical(&logits, &mut rng) as usize;
            counts[idx] += 1;
        }
        for &c in &counts {
            assert!(c > 500, "bucket count {} too low", c);
            assert!(c < 1500, "bucket count {} too high", c);
        }
    }

    #[test]
    fn test_log_probs() {
        let logits = vec![1.0, 2.0, 3.0];
        let lp = compute_log_probs(&logits);
        // log_probs should sum to ~0 in exp space (i.e. exp(lp).sum() ≈ 1.0)
        let sum: f32 = lp.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_n_indices() {
        let vals = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let top2 = top_n_indices(&vals, 2);
        assert_eq!(top2, vec![1, 3]); // indices of 5.0 and 4.0
    }
}
