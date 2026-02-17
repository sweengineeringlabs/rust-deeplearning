//! Mixture-of-Experts layer: router gate + top-k expert selection.

use crate::api::error::NnResult;
use crate::core::feed_forward::FeedForward;
use crate::core::linear::Linear;
use rustml_core::Tensor;

/// Mixture-of-Experts (MoE) layer used in Mixtral-style models.
///
/// Routes each token to `num_experts_per_tok` experts (top-k gating),
/// computes a weighted sum of expert outputs.
pub struct MoeLayer {
    /// Router gate: [d_model, num_experts]
    pub gate: Linear,
    /// One SwiGLU FFN per expert
    pub experts: Vec<FeedForward>,
    /// Number of experts to activate per token (top-k)
    pub num_experts_per_tok: usize,
}

impl MoeLayer {
    /// Create a new MoE layer with random initialization.
    pub fn new(
        d_model: usize,
        hidden_dim: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        bias: bool,
    ) -> Self {
        let gate = Linear::new_no_bias(d_model, num_experts);
        let experts = (0..num_experts)
            .map(|_| FeedForward::swiglu(d_model, hidden_dim, bias))
            .collect();
        Self {
            gate,
            experts,
            num_experts_per_tok,
        }
    }

    /// Construct from pre-loaded weights.
    pub fn from_weights(
        gate: Linear,
        experts: Vec<FeedForward>,
        num_experts_per_tok: usize,
    ) -> Self {
        Self {
            gate,
            experts,
            num_experts_per_tok,
        }
    }

    /// Returns (total_params, frozen_params).
    pub fn parameter_count(&self) -> (usize, usize) {
        let (mut total, mut frozen) = self.gate.parameter_count();
        for expert in &self.experts {
            let (t, f) = expert.parameter_count();
            total += t;
            frozen += f;
        }
        (total, frozen)
    }

    /// Toggle native Q4 integer matmul on gate and all expert FFNs.
    pub fn set_native_q4_matmul(&mut self, enabled: bool) {
        self.gate.set_native_q4_matmul(enabled);
        for expert in &mut self.experts {
            expert.set_native_q4_matmul(enabled);
        }
    }

    /// Forward pass: route each token to top-k experts and merge outputs.
    ///
    /// Input: [B, S, D] → Output: [B, S, D]
    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        let shape = input.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];
        let num_tokens = batch_size * seq_len;

        // Flatten to [B*S, D]
        let flat = input.reshape(&[num_tokens, d_model])?;

        // Router logits: [B*S, num_experts]
        let logits = self.gate.forward(&flat)?;
        let probs = logits.softmax(-1)?;

        // Process each token: top-k expert selection
        let mut output_data = vec![0.0f32; num_tokens * d_model];

        for t in 0..num_tokens {
            // Get router probabilities for this token
            let token_probs: Vec<f32> = (0..self.experts.len())
                .map(|e| probs.get(&[t, e]).unwrap_or(0.0))
                .collect();

            // Find top-k experts by sorting indices
            let mut indices: Vec<usize> = (0..self.experts.len()).collect();
            indices.sort_by(|&a, &b| {
                token_probs[b]
                    .partial_cmp(&token_probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top_k = &indices[..self.num_experts_per_tok];

            // Renormalize weights for selected experts
            let weight_sum: f32 = top_k.iter().map(|&i| token_probs[i]).sum();

            // Get token input: [1, D]
            let token_input = flat.slice(0, t, t + 1)?;

            // Compute weighted sum of expert outputs
            for &expert_idx in top_k {
                let w = if weight_sum > 0.0 {
                    token_probs[expert_idx] / weight_sum
                } else {
                    1.0 / self.num_experts_per_tok as f32
                };
                let expert_out = self.experts[expert_idx].forward(&token_input)?;
                let expert_data: Vec<f32> = expert_out.iter().collect();
                for d in 0..d_model {
                    output_data[t * d_model + d] += w * expert_data[d];
                }
            }
        }

        Tensor::from_vec(output_data, vec![batch_size, seq_len, d_model])
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_layer_shape() {
        let moe = MoeLayer::new(64, 256, 4, 2, false);
        let x = Tensor::randn(vec![1, 8, 64]);
        let y = moe.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_moe_layer_parameter_count() {
        let moe = MoeLayer::new(64, 256, 4, 2, false);
        let (total, frozen) = moe.parameter_count();
        // gate: 64*4 = 256
        // 4 experts, each SwiGLU: up(64*256) + gate(64*256) + down(256*64) = 3*16384 = 49152
        // total = 256 + 4*49152 = 256 + 196608 = 196864
        assert_eq!(total, 256 + 4 * 3 * 64 * 256);
        assert_eq!(frozen, 0);
    }
}
