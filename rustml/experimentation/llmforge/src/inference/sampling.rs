use rand::Rng;

/// Return the index of the maximum value in the logit slice.
pub(crate) fn argmax(logits: &[f32]) -> u32 {
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
pub(crate) fn apply_repetition_penalty(logits: &mut [f32], past_tokens: &[u32], penalty: f32) {
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
pub(crate) fn apply_top_k(logits: &mut [f32], k: usize) {
    if k >= logits.len() {
        return;
    }
    // Find the k-th largest value using partial sort.
    let mut vals: Vec<f32> = logits.to_vec();
    // select_nth_unstable_by puts the k-th largest at position (len - k)
    let pivot = vals.len() - k;
    vals.select_nth_unstable_by(pivot, |a: &f32, b: &f32| a.total_cmp(b));
    let threshold = vals[pivot];

    // Count how many values are >= threshold (might be more than k due to ties)
    // We keep all values >= threshold to avoid masking ties arbitrarily
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Apply nucleus (top-p) sampling: keep the smallest set of tokens whose
/// cumulative probability exceeds `p`, mask the rest to NEG_INFINITY.
/// If p >= 1.0, this is a no-op.
pub(crate) fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }

    // Stable softmax to get probabilities
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Sort indices by probability descending
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

    // Cumulative sum; find cutoff
    let mut cumsum = 0.0f32;
    let mut keep = vec![false; logits.len()];
    for &idx in &indices {
        cumsum += probs[idx];
        keep[idx] = true;
        if cumsum >= p {
            break;
        }
    }

    // Mask tokens not in the nucleus
    for (i, v) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Sample from a categorical distribution defined by logits (not probabilities).
/// Applies stable softmax internally then draws using the given RNG.
pub(crate) fn sample_categorical<R: Rng>(logits: &[f32], rng: &mut R) -> u32 {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
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
        // Top-2 values are 5.0 and 4.0
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count <= 3); // at most k + ties
        assert!(finite_count >= 2);
        // The top values should still be present
        assert!(logits[1].is_finite()); // 5.0
        assert!(logits[4].is_finite()); // 4.0
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_top_k(&mut logits, 10);
        assert_eq!(logits, original); // no-op
    }

    #[test]
    fn test_top_p_keeps_minimum_set() {
        // One dominant token with prob ~1.0
        let mut logits = vec![10.0, -10.0, -10.0, -10.0];
        apply_top_p(&mut logits, 0.9);
        // The dominant token (idx 0) should be kept
        assert!(logits[0].is_finite());
        // Others should be masked
        for &v in &logits[1..] {
            assert_eq!(v, f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_top_p_one_keeps_all() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let original = logits.clone();
        apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, original); // no-op
    }

    #[test]
    fn test_repetition_penalty_positive_logits() {
        let mut logits = vec![2.0, 4.0, 6.0];
        apply_repetition_penalty(&mut logits, &[0, 2], 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6); // 2.0 / 2.0
        assert!((logits[1] - 4.0).abs() < 1e-6); // unchanged
        assert!((logits[2] - 3.0).abs() < 1e-6); // 6.0 / 2.0
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut logits = vec![-2.0, 1.0, -3.0];
        apply_repetition_penalty(&mut logits, &[0, 2], 2.0);
        assert!((logits[0] - (-4.0)).abs() < 1e-6); // -2.0 * 2.0
        assert!((logits[1] - 1.0).abs() < 1e-6);    // unchanged
        assert!((logits[2] - (-6.0)).abs() < 1e-6); // -3.0 * 2.0
    }

    #[test]
    fn test_sample_categorical_deterministic() {
        // Only one valid token (rest are NEG_INFINITY)
        let logits = vec![f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            assert_eq!(sample_categorical(&logits, &mut rng), 1);
        }
    }

    #[test]
    fn test_sample_categorical_distribution() {
        // Uniform logits should produce roughly uniform samples
        let logits = vec![0.0; 4];
        let mut rng = rand::thread_rng();
        let mut counts = [0u32; 4];
        let n = 4000;
        for _ in 0..n {
            let idx = sample_categorical(&logits, &mut rng) as usize;
            counts[idx] += 1;
        }
        // Each bucket should get roughly n/4 = 1000; allow wide margin
        for &c in &counts {
            assert!(c > 500, "bucket count {} too low", c);
            assert!(c < 1500, "bucket count {} too high", c);
        }
    }
}
