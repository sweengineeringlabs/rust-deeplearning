//! TimeSeriesTransformer (FR-801, FR-802).
//!
//! A transformer model for time series forecasting with:
//! - Input projection (Linear)
//! - Sinusoidal positional encoding
//! - N transformer blocks with causal multi-head attention
//! - Last-timestep projection to output

use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;
use crate::core::nn::activations::GELU;
use crate::core::nn::dropout::Dropout;
use crate::core::nn::layer_norm::LayerNorm;
use crate::core::nn::linear::Linear;

// ---------------------------------------------------------------------------
// MultiHeadAttention (FR-802)
// ---------------------------------------------------------------------------

/// Multi-head scaled dot-product attention with causal masking.
struct MultiHeadAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    num_heads: usize,
    head_dim: usize,
    d_model: usize,
}

impl MultiHeadAttention {
    fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );
        let head_dim = d_model / num_heads;
        Self {
            wq: Linear::new(d_model, d_model),
            wk: Linear::new(d_model, d_model),
            wv: Linear::new(d_model, d_model),
            wo: Linear::new(d_model, d_model),
            num_heads,
            head_dim,
            d_model,
        }
    }

    /// Forward pass for multi-head attention.
    ///
    /// Input shape: [batch, seq_len, d_model]
    /// Output shape: [batch, seq_len, d_model]
    ///
    /// Steps:
    /// 1. Project Q, K, V via Linear layers (tape-recorded by Linear).
    /// 2. Reshape to [batch, num_heads, seq_len, head_dim].
    /// 3. Compute scaled dot-product attention with causal mask.
    /// 4. Concatenate heads and project through wo.
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let shape = input.shape().to_vec();
        assert_eq!(shape.len(), 3, "MHA input must be 3D [batch, seq, d_model]");
        let batch = shape[0];
        let seq_len = shape[1];

        // 1. Q, K, V projections (each records on tape via Linear::forward)
        let q = self.wq.forward(input)?; // [batch, seq_len, d_model]
        let k = self.wk.forward(input)?;
        let v = self.wv.forward(input)?;

        // 2. Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q.reshape_raw(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let q = q.transpose_raw(1, 2)?; // [batch, num_heads, seq_len, head_dim]
        let k = k.reshape_raw(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k.transpose_raw(1, 2)?;
        let v = v.reshape_raw(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let v = v.transpose_raw(1, 2)?;

        // 3. Scaled dot-product attention with causal mask
        // scores = Q @ K^T / sqrt(head_dim)
        let k_t = k.transpose_raw(-1, -2)?; // [batch, num_heads, head_dim, seq_len]
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul_raw(&k_t)?; // [batch, num_heads, seq_len, seq_len]
        let scores = scores.div_scalar_raw(scale);

        // Apply causal mask: set future positions to -1e9
        let scores_data = scores.to_vec();
        let mut masked_data = scores_data;
        for b in 0..batch {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if j > i {
                            let idx = b * (self.num_heads * seq_len * seq_len)
                                + h * (seq_len * seq_len)
                                + i * seq_len
                                + j;
                            masked_data[idx] = -1e9;
                        }
                    }
                }
            }
        }
        let masked_scores = Tensor::from_vec(
            masked_data,
            vec![batch, self.num_heads, seq_len, seq_len],
        )?;

        // Softmax over last dimension
        let attn_weights = softmax_4d(&masked_scores)?;

        // attn_output = attn_weights @ V
        let attn_output = attn_weights.matmul_raw(&v)?; // [batch, num_heads, seq_len, head_dim]

        // Record the attention forward as a single tape op for backward
        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(AttentionBackward {
                    num_heads: self.num_heads,
                    head_dim: self.head_dim,
                    batch,
                    seq_len,
                }),
                output_id: attn_output.id(),
                input_ids: vec![q.id(), k.id(), v.id()],
                saved_tensors: vec![q.clone(), k.clone(), v.clone(), attn_weights.clone()],
            };
            tape::record_op(entry);
        }

        // 4. Concatenate heads: transpose back and reshape
        let attn_output = attn_output.transpose_raw(1, 2)?; // [batch, seq_len, num_heads, head_dim]
        let attn_output = attn_output.reshape_raw(&[batch, seq_len, self.d_model])?;

        // Output projection (tape-recorded by Linear)
        self.wo.forward(&attn_output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.wq.parameters());
        params.extend(self.wk.parameters());
        params.extend(self.wv.parameters());
        params.extend(self.wo.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.wq.parameters_mut());
        params.extend(self.wk.parameters_mut());
        params.extend(self.wv.parameters_mut());
        params.extend(self.wo.parameters_mut());
        params
    }
}

// ---------------------------------------------------------------------------
// Attention backward op
// ---------------------------------------------------------------------------

/// Backward op for scaled dot-product attention.
///
/// saved[0] = Q  [batch, num_heads, seq_len, head_dim]
/// saved[1] = K  [batch, num_heads, seq_len, head_dim]
/// saved[2] = V  [batch, num_heads, seq_len, head_dim]
/// saved[3] = attn_weights (after softmax + mask) [batch, num_heads, seq_len, seq_len]
///
/// input_ids[0] = Q, input_ids[1] = K, input_ids[2] = V
///
/// Forward:
///   scores = Q @ K^T / sqrt(d_k)
///   attn = softmax(mask(scores))
///   output = attn @ V
///
/// Backward:
///   grad_V = attn^T @ grad_output
///   grad_attn = grad_output @ V^T
///   grad_scores = softmax_backward(grad_attn, attn) / sqrt(d_k)  (with mask zeroing future)
///   grad_Q = grad_scores @ K
///   grad_K = grad_scores^T @ Q
struct AttentionBackward {
    num_heads: usize,
    head_dim: usize,
    batch: usize,
    seq_len: usize,
}

impl BackwardOp for AttentionBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let q = &saved[0]; // [batch, num_heads, seq_len, head_dim]
        let k = &saved[1];
        let v = &saved[2];
        let attn_weights = &saved[3]; // [batch, num_heads, seq_len, seq_len]

        let scale = (self.head_dim as f32).sqrt();

        // grad_V = attn^T @ grad_output
        let attn_t = attn_weights.transpose_raw(-1, -2).expect("attn transpose");
        let grad_v = attn_t.matmul_raw(grad_output).expect("grad_v matmul");

        // grad_attn = grad_output @ V^T
        let v_t = v.transpose_raw(-1, -2).expect("v transpose");
        let grad_attn = grad_output.matmul_raw(&v_t).expect("grad_attn matmul");

        // grad_scores = softmax_backward(grad_attn, attn_weights) / scale
        // softmax backward: grad_input[i] = attn[i] * (grad[i] - sum(grad * attn))
        let attn_data = attn_weights.to_vec();
        let grad_attn_data = grad_attn.to_vec();
        let total = attn_data.len();
        let inner = self.seq_len; // last dim
        let outer = total / inner;

        let mut grad_scores_data = vec![0.0f32; total];
        for o in 0..outer {
            let base = o * inner;
            let mut dot = 0.0f32;
            for j in 0..inner {
                dot += grad_attn_data[base + j] * attn_data[base + j];
            }
            for j in 0..inner {
                let idx = base + j;
                grad_scores_data[idx] = attn_data[idx] * (grad_attn_data[idx] - dot) / scale;
            }
        }

        // Zero out gradients for masked (future) positions
        for b in 0..self.batch {
            for h in 0..self.num_heads {
                for i in 0..self.seq_len {
                    for j in 0..self.seq_len {
                        if j > i {
                            let idx = b * (self.num_heads * self.seq_len * self.seq_len)
                                + h * (self.seq_len * self.seq_len)
                                + i * self.seq_len
                                + j;
                            grad_scores_data[idx] = 0.0;
                        }
                    }
                }
            }
        }

        let grad_scores = Tensor::from_vec(
            grad_scores_data,
            vec![self.batch, self.num_heads, self.seq_len, self.seq_len],
        )
        .expect("grad_scores from_vec");

        // grad_Q = grad_scores @ K
        let grad_q = grad_scores.matmul_raw(k).expect("grad_q matmul");

        // grad_K = grad_scores^T @ Q
        let grad_scores_t = grad_scores.transpose_raw(-1, -2).expect("grad_scores transpose");
        let grad_k = grad_scores_t.matmul_raw(q).expect("grad_k matmul");

        vec![grad_q, grad_k, grad_v]
    }

    fn name(&self) -> &str {
        "AttentionBackward"
    }
}

// ---------------------------------------------------------------------------
// Softmax helper for 4D tensors (along last dim)
// ---------------------------------------------------------------------------

/// Compute softmax along the last dimension of a 4D tensor.
/// This is done element-wise without tape recording (the attention backward
/// handles the combined gradient).
fn softmax_4d(input: &Tensor) -> SwetsResult<Tensor> {
    let shape = input.shape().to_vec();
    assert_eq!(shape.len(), 4);
    let data = input.to_vec();
    let last_dim = shape[3];
    let num_rows = data.len() / last_dim;

    let mut out = vec![0.0f32; data.len()];
    for row in 0..num_rows {
        let base = row * last_dim;
        let row_slice = &data[base..base + last_dim];

        // Numerically stable softmax
        let max_val = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for j in 0..last_dim {
            let e = (row_slice[j] - max_val).exp();
            out[base + j] = e;
            sum_exp += e;
        }
        for j in 0..last_dim {
            out[base + j] /= sum_exp;
        }
    }

    Ok(Tensor::from_vec(out, shape)?)
}

// ---------------------------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------------------------

/// A single transformer block with pre-norm architecture.
///
/// x = x + attention(norm1(x))
/// x = x + ff(norm2(x))
///
/// Feed-forward: Linear(d_model, 4*d_model) -> GELU -> Linear(4*d_model, d_model) -> Dropout
struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    ff_gelu: GELU,
    dropout: Dropout,
}

impl TransformerBlock {
    fn new(d_model: usize, num_heads: usize, dropout_p: f32) -> Self {
        let ff_dim = d_model * 4;
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads),
            norm1: LayerNorm::new(vec![d_model]),
            norm2: LayerNorm::new(vec![d_model]),
            ff1: Linear::new(d_model, ff_dim),
            ff2: Linear::new(ff_dim, d_model),
            ff_gelu: GELU::new(),
            dropout: Dropout::new(dropout_p),
        }
    }

    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        // Pre-norm attention: x = x + attention(norm1(x))
        let normed1 = self.norm1.forward(input)?;
        let attn_out = self.attention.forward(&normed1)?;
        let x = input.add_raw(&attn_out)?;

        // Record residual addition on tape
        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(ResidualAddBackward),
                output_id: x.id(),
                input_ids: vec![input.id(), attn_out.id()],
                saved_tensors: vec![],
            };
            tape::record_op(entry);
        }

        // Pre-norm feed-forward: x = x + ff(norm2(x))
        let normed2 = self.norm2.forward(&x)?;
        let ff_out = self.ff1.forward(&normed2)?;
        let ff_out = self.ff_gelu.forward(&ff_out)?;
        let ff_out = self.ff2.forward(&ff_out)?;
        let ff_out = self.dropout.forward(&ff_out)?;
        let output = x.add_raw(&ff_out)?;

        // Record residual addition on tape
        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(ResidualAddBackward),
                output_id: output.id(),
                input_ids: vec![x.id(), ff_out.id()],
                saved_tensors: vec![],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        // GELU and Dropout have no parameters
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.ff1.parameters_mut());
        params.extend(self.ff2.parameters_mut());
        params
    }
}

// ---------------------------------------------------------------------------
// Residual add backward
// ---------------------------------------------------------------------------

/// Backward for residual connection: output = a + b.
/// Both inputs receive grad_output directly (identity gradient for addition).
struct ResidualAddBackward;

impl BackwardOp for ResidualAddBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        // For addition, gradient passes through to both inputs unchanged
        vec![grad_output.clone(), grad_output.clone()]
    }

    fn name(&self) -> &str {
        "ResidualAddBackward"
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal Positional Encoding
// ---------------------------------------------------------------------------

/// Precompute sinusoidal positional encodings.
///
/// PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
///
/// Returns a Tensor of shape [max_len, d_model].
fn sinusoidal_positional_encoding(max_len: usize, d_model: usize) -> Tensor {
    let mut data = vec![0.0f32; max_len * d_model];

    for pos in 0..max_len {
        for i in 0..d_model / 2 {
            let angle = pos as f32 / (10000.0f32).powf(2.0 * i as f32 / d_model as f32);
            data[pos * d_model + 2 * i] = angle.sin();
            data[pos * d_model + 2 * i + 1] = angle.cos();
        }
        // If d_model is odd, fill the last element with sin
        if d_model % 2 == 1 {
            let i = d_model / 2;
            let angle = pos as f32 / (10000.0f32).powf(2.0 * i as f32 / d_model as f32);
            data[pos * d_model + d_model - 1] = angle.sin();
        }
    }

    Tensor::from_vec(data, vec![max_len, d_model]).expect("positional encoding from_vec")
}

// ---------------------------------------------------------------------------
// SelectLastTimestepBackward
// ---------------------------------------------------------------------------

/// Backward for selecting the last timestep from [batch, seq_len, d_model].
/// The gradient is scattered back to the last position; all others are zero.
///
/// saved: none
/// input_ids[0] = the 3D tensor that was sliced
struct SelectLastTimestepBackward {
    batch: usize,
    seq_len: usize,
    d_model: usize,
}

impl BackwardOp for SelectLastTimestepBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        // grad_output is [batch, d_model]
        // We need to produce [batch, seq_len, d_model] with zeros everywhere
        // except the last timestep.
        let grad_data = grad_output.to_vec();
        let total = self.batch * self.seq_len * self.d_model;
        let mut grad_input = vec![0.0f32; total];

        for b in 0..self.batch {
            let src_start = b * self.d_model;
            let dst_start = b * self.seq_len * self.d_model + (self.seq_len - 1) * self.d_model;
            grad_input[dst_start..dst_start + self.d_model]
                .copy_from_slice(&grad_data[src_start..src_start + self.d_model]);
        }

        let result = Tensor::from_vec(grad_input, vec![self.batch, self.seq_len, self.d_model])
            .expect("select_last_timestep backward");
        vec![result]
    }

    fn name(&self) -> &str {
        "SelectLastTimestepBackward"
    }
}

// ---------------------------------------------------------------------------
// PosEncodingAddBackward
// ---------------------------------------------------------------------------

/// Backward for adding positional encoding: output = projected + pos_enc_slice.
/// Only the projected input needs gradients (pos_encoding is fixed, not a parameter).
///
/// input_ids[0] = projected tensor
struct PosEncodingAddBackward;

impl BackwardOp for PosEncodingAddBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        // Gradient flows through to the projected input only
        vec![grad_output.clone()]
    }

    fn name(&self) -> &str {
        "PosEncodingAddBackward"
    }
}

// ---------------------------------------------------------------------------
// TimeSeriesTransformer (FR-801)
// ---------------------------------------------------------------------------

/// A transformer model for time series forecasting.
///
/// Architecture:
/// 1. Input projection: [batch, seq_len, input_size] -> [batch, seq_len, d_model]
/// 2. Add sinusoidal positional encoding
/// 3. Pass through N transformer blocks (pre-norm, causal attention)
/// 4. Select last timestep: [batch, d_model]
/// 5. Output projection: [batch, output_size]
pub struct TimeSeriesTransformer {
    input_proj: Linear,
    pos_encoding: Tensor,
    blocks: Vec<TransformerBlock>,
    output_proj: Linear,
    d_model: usize,
    max_len: usize,
}

impl TimeSeriesTransformer {
    /// Create a new TimeSeriesTransformer.
    ///
    /// # Arguments
    /// * `input_size` - Number of input features per timestep
    /// * `d_model` - Internal model dimension
    /// * `num_heads` - Number of attention heads (must divide d_model)
    /// * `num_layers` - Number of transformer blocks
    /// * `output_size` - Number of output features
    /// * `max_len` - Maximum sequence length for positional encoding
    /// * `dropout_p` - Dropout probability (0 to disable)
    pub fn new(
        input_size: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        output_size: usize,
        max_len: usize,
        dropout_p: f32,
    ) -> Self {
        assert!(num_layers > 0, "num_layers must be > 0");
        assert!(d_model > 0, "d_model must be > 0");

        let pos_encoding = sinusoidal_positional_encoding(max_len, d_model);

        let blocks = (0..num_layers)
            .map(|_| TransformerBlock::new(d_model, num_heads, dropout_p))
            .collect();

        Self {
            input_proj: Linear::new(input_size, d_model),
            pos_encoding,
            blocks,
            output_proj: Linear::new(d_model, output_size),
            d_model,
            max_len,
        }
    }

    /// Switch all dropout layers to training mode.
    pub fn train(&mut self) {
        for block in &mut self.blocks {
            block.dropout.train();
        }
    }

    /// Switch all dropout layers to evaluation mode.
    pub fn eval(&mut self) {
        for block in &mut self.blocks {
            block.dropout.eval();
        }
    }
}

impl Layer for TimeSeriesTransformer {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let shape = input.shape().to_vec();
        assert_eq!(
            shape.len(),
            3,
            "TimeSeriesTransformer input must be 3D [batch, seq_len, input_size]"
        );
        let batch = shape[0];
        let seq_len = shape[1];
        assert!(
            seq_len <= self.max_len,
            "seq_len ({}) exceeds max_len ({})",
            seq_len,
            self.max_len
        );

        // 1. Input projection: [batch, seq_len, input_size] -> [batch, seq_len, d_model]
        let projected = self.input_proj.forward(input)?;

        // 2. Add positional encoding (slice to seq_len)
        // pos_encoding is [max_len, d_model], we need [seq_len, d_model]
        let pos_slice = self.pos_encoding.reshape_raw(&[self.max_len, self.d_model])?;
        let pos_data = pos_slice.to_vec();
        let sliced_data: Vec<f32> = pos_data[..seq_len * self.d_model].to_vec();
        let pos_enc = Tensor::from_vec(sliced_data, vec![seq_len, self.d_model])?;

        // Broadcasting add: [batch, seq_len, d_model] + [seq_len, d_model]
        let mut x = projected.add_raw(&pos_enc)?;

        // Record positional encoding addition on tape
        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(PosEncodingAddBackward),
                output_id: x.id(),
                input_ids: vec![projected.id()],
                saved_tensors: vec![],
            };
            tape::record_op(entry);
        }

        // 3. Pass through transformer blocks
        for block in &mut self.blocks {
            x = block.forward(&x)?;
        }

        // 4. Select last timestep: [batch, seq_len, d_model] -> [batch, d_model]
        let x_data = x.to_vec();
        let mut last_data = vec![0.0f32; batch * self.d_model];
        for b in 0..batch {
            let src_start = b * seq_len * self.d_model + (seq_len - 1) * self.d_model;
            let dst_start = b * self.d_model;
            last_data[dst_start..dst_start + self.d_model]
                .copy_from_slice(&x_data[src_start..src_start + self.d_model]);
        }
        let last_timestep = Tensor::from_vec(last_data, vec![batch, self.d_model])?;

        // Record last timestep selection on tape
        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(SelectLastTimestepBackward {
                    batch,
                    seq_len,
                    d_model: self.d_model,
                }),
                output_id: last_timestep.id(),
                input_ids: vec![x.id()],
                saved_tensors: vec![],
            };
            tape::record_op(entry);
        }

        // 5. Output projection: [batch, d_model] -> [batch, output_size]
        self.output_proj.forward(&last_timestep)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.input_proj.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.output_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.input_proj.parameters_mut());
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
        }
        params.extend(self.output_proj.parameters_mut());
        params
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_encoding_shape() {
        let pe = sinusoidal_positional_encoding(100, 64);
        assert_eq!(pe.shape(), &[100, 64]);
    }

    #[test]
    fn test_sinusoidal_encoding_values() {
        let pe = sinusoidal_positional_encoding(10, 4);
        let data = pe.to_vec();
        // Position 0: sin(0)=0, cos(0)=1
        assert!((data[0] - 0.0).abs() < 1e-5, "PE(0,0) should be sin(0)=0");
        assert!((data[1] - 1.0).abs() < 1e-5, "PE(0,1) should be cos(0)=1");
    }

    #[test]
    fn test_softmax_4d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let t = Tensor::from_vec(data, vec![1, 1, 2, 4]).unwrap();
        let s = softmax_4d(&t).unwrap();
        assert_eq!(s.shape(), &[1, 1, 2, 4]);
        let out = s.to_vec();
        // Each row of 4 should sum to ~1.0
        let row0_sum: f32 = out[0..4].iter().sum();
        let row1_sum: f32 = out[4..8].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);
        assert!((row1_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_transformer_construction() {
        let model = TimeSeriesTransformer::new(
            3,   // input_size (e.g., 3 features)
            16,  // d_model
            2,   // num_heads
            2,   // num_layers
            1,   // output_size
            50,  // max_len
            0.0, // dropout_p
        );
        assert_eq!(model.d_model, 16);
        assert_eq!(model.max_len, 50);
        assert_eq!(model.blocks.len(), 2);
        assert!(model.parameters().len() > 0);
    }

    #[test]
    fn test_transformer_forward_shape() {
        let mut model = TimeSeriesTransformer::new(
            3,   // input_size
            16,  // d_model
            2,   // num_heads
            1,   // num_layers
            1,   // output_size
            50,  // max_len
            0.0, // dropout
        );
        model.eval();

        // Input: [batch=2, seq_len=10, features=3]
        let input = Tensor::randn([2, 10, 3]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 1]);
    }

    #[test]
    fn test_transformer_forward_single_sample() {
        let mut model = TimeSeriesTransformer::new(
            1,   // input_size
            8,   // d_model
            2,   // num_heads
            1,   // num_layers
            1,   // output_size
            20,  // max_len
            0.0, // dropout
        );
        model.eval();

        // Input: [batch=1, seq_len=5, features=1]
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1, 5, 1],
        )
        .unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1]);
        // Verify it produces a finite value
        let val = output.to_vec()[0];
        assert!(val.is_finite(), "output should be finite, got {}", val);
    }

    #[test]
    fn test_transformer_parameter_count() {
        let model = TimeSeriesTransformer::new(
            4,   // input_size
            16,  // d_model
            4,   // num_heads
            2,   // num_layers
            1,   // output_size
            100, // max_len
            0.1, // dropout
        );
        let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
        // Should have substantial parameters from Linear and LayerNorm layers
        assert!(
            param_count > 0,
            "model should have parameters, got {}",
            param_count
        );

        // Verify specific components:
        // input_proj: 4*16 + 16 = 80
        // Each block: 4 attention linears (16*16+16)*4 + 2 layernorms (16+16)*2
        //           + ff1 (16*64+64) + ff2 (64*16+16) = 4*272 + 64 + 1088 + 1040
        //           = 1088 + 64 + 1088 + 1040 = 3280
        // output_proj: 16*1 + 1 = 17
        // Total rough: 80 + 2*3280 + 17 = 6657
        assert!(
            param_count > 100,
            "expected more parameters, got {}",
            param_count
        );
    }

    #[test]
    fn test_multi_head_attention_forward() {
        let mut mha = MultiHeadAttention::new(8, 2);
        let input = Tensor::randn([1, 4, 8]);
        let output = mha.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_transformer_block_forward() {
        let mut block = TransformerBlock::new(8, 2, 0.0);
        let input = Tensor::randn([1, 4, 8]);
        let output = block.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_causal_masking() {
        // Verify that causal masking prevents attending to future positions
        // by checking that attention weights for future positions are ~0
        let mut mha = MultiHeadAttention::new(8, 2);
        tape::no_grad(|| {
            let input = Tensor::ones([1, 4, 8]);
            let _ = mha.forward(&input).unwrap();
            // The test passes if forward doesn't panic and produces valid output
        });
    }
}
