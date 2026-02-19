use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;

// ---------------------------------------------------------------------------
// Backward ops
// ---------------------------------------------------------------------------

/// Backward for a single causal dilated 1D convolution.
///
/// Forward: output[b][o][t] = bias[o] + sum_i sum_k weight[o][i][k] * padded_input[b][i][t + k * dilation]
///
/// saved[0] = padded_input  [batch, in_ch, padded_len]
/// saved[1] = weight        [out_ch, in_ch, kernel_size]
/// saved[2] = bias          [out_ch]
///
/// input_ids[0] = original input (pre-padding) id
/// input_ids[1] = weight id
/// input_ids[2] = bias id
struct CausalConv1dBackward {
    /// Shape of the *original* input before causal padding: [batch, in_ch, seq_len]
    input_shape: Vec<usize>,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
    padding: usize,
}

impl BackwardOp for CausalConv1dBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let padded_input = &saved[0];
        let weight = &saved[1];
        // saved[2] = bias (not needed for its own grad computation aside from
        // summing grad_output, which we do below)

        let grad_data = grad_output.to_vec();
        let pad_data = padded_input.to_vec();
        let w_data = weight.to_vec();

        let batch = self.input_shape[0];
        let in_ch = self.input_shape[1];
        let seq_len = self.input_shape[2];
        let out_ch = self.out_channels;
        let ks = self.kernel_size;
        let dil = self.dilation;
        let pad = self.padding;
        let padded_len = seq_len + pad;
        let out_len = seq_len; // causal conv preserves length

        // --- grad_weight: dL/dW[o][i][k] = sum_b sum_t grad_output[b][o][t] * padded_input[b][i][t + k*dil]
        let mut grad_weight = vec![0.0f32; out_ch * in_ch * ks];
        for b_idx in 0..batch {
            for o in 0..out_ch {
                for t in 0..out_len {
                    let go = grad_data[b_idx * out_ch * out_len + o * out_len + t];
                    for i in 0..in_ch {
                        for k in 0..ks {
                            let p_idx = t + k * dil;
                            if p_idx < padded_len {
                                let pv = pad_data[b_idx * in_ch * padded_len + i * padded_len + p_idx];
                                grad_weight[o * in_ch * ks + i * ks + k] += go * pv;
                            }
                        }
                    }
                }
            }
        }

        // --- grad_bias: dL/db[o] = sum_b sum_t grad_output[b][o][t]
        let mut grad_bias = vec![0.0f32; out_ch];
        for b_idx in 0..batch {
            for o in 0..out_ch {
                for t in 0..out_len {
                    grad_bias[o] += grad_data[b_idx * out_ch * out_len + o * out_len + t];
                }
            }
        }

        // --- grad_input (original, before padding)
        // dL/d padded_input[b][i][p] = sum_o sum_{k where p = t + k*dil} grad_output[b][o][t] * weight[o][i][k]
        //   which is equivalent to:
        //   for each (o, t, k): padded_input index = t + k*dil -> accumulate into that index
        // Then we strip the left-padding to get grad for original input.
        let mut grad_padded = vec![0.0f32; batch * in_ch * padded_len];
        for b_idx in 0..batch {
            for o in 0..out_ch {
                for t in 0..out_len {
                    let go = grad_data[b_idx * out_ch * out_len + o * out_len + t];
                    for i in 0..in_ch {
                        for k in 0..ks {
                            let p_idx = t + k * dil;
                            if p_idx < padded_len {
                                let w_val = w_data[o * in_ch * ks + i * ks + k];
                                grad_padded[b_idx * in_ch * padded_len + i * padded_len + p_idx] +=
                                    go * w_val;
                            }
                        }
                    }
                }
            }
        }

        // Strip left padding to get grad_input [batch, in_ch, seq_len]
        let mut grad_input = vec![0.0f32; batch * in_ch * seq_len];
        for b_idx in 0..batch {
            for i in 0..in_ch {
                for t in 0..seq_len {
                    grad_input[b_idx * in_ch * seq_len + i * seq_len + t] =
                        grad_padded[b_idx * in_ch * padded_len + i * padded_len + pad + t];
                }
            }
        }

        let grad_input_t =
            Tensor::from_vec(grad_input, self.input_shape.clone()).expect("conv1d grad_input");
        let grad_weight_t =
            Tensor::from_vec(grad_weight, vec![out_ch, in_ch, ks]).expect("conv1d grad_weight");
        let grad_bias_t =
            Tensor::from_vec(grad_bias, vec![out_ch]).expect("conv1d grad_bias");

        vec![grad_input_t, grad_weight_t, grad_bias_t]
    }

    fn name(&self) -> &str {
        "CausalConv1dBackward"
    }
}

/// Backward for a 1x1 convolution used as residual projection.
///
/// Forward: output[b][o][t] = bias[o] + sum_i weight[o][i] * input[b][i][t]
///
/// saved[0] = input   [batch, in_ch, seq_len]
/// saved[1] = weight  [out_ch, in_ch, 1]  (stored as [out_ch, in_ch, 1])
///
/// input_ids[0] = input id, input_ids[1] = weight id, input_ids[2] = bias id
struct Residual1x1Backward {
    input_shape: Vec<usize>,
    out_channels: usize,
}

impl BackwardOp for Residual1x1Backward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let weight = &saved[1];

        let grad_data = grad_output.to_vec();
        let in_data = input.to_vec();
        let w_data = weight.to_vec();

        let batch = self.input_shape[0];
        let in_ch = self.input_shape[1];
        let seq_len = self.input_shape[2];
        let out_ch = self.out_channels;

        // grad_weight[o][i] = sum_b sum_t grad_output[b][o][t] * input[b][i][t]
        let mut grad_weight = vec![0.0f32; out_ch * in_ch];
        for b_idx in 0..batch {
            for o in 0..out_ch {
                for t in 0..seq_len {
                    let go = grad_data[b_idx * out_ch * seq_len + o * seq_len + t];
                    for i in 0..in_ch {
                        let iv = in_data[b_idx * in_ch * seq_len + i * seq_len + t];
                        grad_weight[o * in_ch + i] += go * iv;
                    }
                }
            }
        }

        // grad_bias[o] = sum_b sum_t grad_output[b][o][t]
        let mut grad_bias = vec![0.0f32; out_ch];
        for b_idx in 0..batch {
            for o in 0..out_ch {
                for t in 0..seq_len {
                    grad_bias[o] += grad_data[b_idx * out_ch * seq_len + o * seq_len + t];
                }
            }
        }

        // grad_input[b][i][t] = sum_o weight[o][i] * grad_output[b][o][t]
        let mut grad_input = vec![0.0f32; batch * in_ch * seq_len];
        for b_idx in 0..batch {
            for i in 0..in_ch {
                for t in 0..seq_len {
                    let mut val = 0.0f32;
                    for o in 0..out_ch {
                        let go = grad_data[b_idx * out_ch * seq_len + o * seq_len + t];
                        // weight stored as [out_ch, in_ch, 1], index = o * in_ch * 1 + i * 1
                        let w_val = w_data[o * in_ch + i];
                        val += go * w_val;
                    }
                    grad_input[b_idx * in_ch * seq_len + i * seq_len + t] = val;
                }
            }
        }

        let grad_input_t =
            Tensor::from_vec(grad_input, self.input_shape.clone()).expect("1x1 grad_input");
        // Reshape grad_weight to [out_ch, in_ch, 1] to match weight shape
        let grad_weight_t =
            Tensor::from_vec(grad_weight, vec![out_ch, in_ch, 1]).expect("1x1 grad_weight");
        let grad_bias_t =
            Tensor::from_vec(grad_bias, vec![out_ch]).expect("1x1 grad_bias");

        vec![grad_input_t, grad_weight_t, grad_bias_t]
    }

    fn name(&self) -> &str {
        "Residual1x1Backward"
    }
}

/// Backward for ReLU applied to 3D tensors.
///
/// saved[0] = pre-activation input
struct ReLU3dBackward;

impl BackwardOp for ReLU3dBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let in_data = input.to_vec();
        let grad_data = grad_output.to_vec();

        let grad_input: Vec<f32> = in_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect();

        let result =
            Tensor::from_vec(grad_input, input.shape().to_vec()).expect("relu3d backward");
        vec![result]
    }

    fn name(&self) -> &str {
        "ReLU3dBackward"
    }
}

/// Backward for element-wise addition of two same-shaped tensors.
struct Add3dBackward;

impl BackwardOp for Add3dBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        // Both inputs have the same shape, so gradients pass through directly
        vec![grad_output.clone(), grad_output.clone()]
    }

    fn name(&self) -> &str {
        "Add3dBackward"
    }
}

/// Backward for global average pooling over the time dimension.
///
/// Forward: output[b][c] = mean_t(input[b][c][t])
/// Backward: grad_input[b][c][t] = grad_output[b][c] / seq_len
///
/// input_ids[0] = input id
struct GlobalAvgPoolBackward {
    input_shape: Vec<usize>, // [batch, channels, seq_len]
}

impl BackwardOp for GlobalAvgPoolBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        let batch = self.input_shape[0];
        let channels = self.input_shape[1];
        let seq_len = self.input_shape[2];
        let go_data = grad_output.to_vec();

        let mut grad_input = vec![0.0f32; batch * channels * seq_len];
        let inv_len = 1.0 / seq_len as f32;

        for b_idx in 0..batch {
            for c in 0..channels {
                let g = go_data[b_idx * channels + c] * inv_len;
                for t in 0..seq_len {
                    grad_input[b_idx * channels * seq_len + c * seq_len + t] = g;
                }
            }
        }

        let result =
            Tensor::from_vec(grad_input, self.input_shape.clone()).expect("gap backward");
        vec![result]
    }

    fn name(&self) -> &str {
        "GlobalAvgPoolBackward"
    }
}

/// Backward for the final fully-connected projection (fc).
///
/// Forward: output[b][o] = sum_c fc_weight[o][c] * pooled[b][c] + fc_bias[o]
///
/// saved[0] = pooled  [batch, channels]
/// saved[1] = fc_weight [output_size, channels]
///
/// input_ids[0] = pooled id, input_ids[1] = fc_weight id, input_ids[2] = fc_bias id
struct FcProjectionBackward {
    pooled_shape: Vec<usize>,    // [batch, channels]
    bias_shape: Vec<usize>,      // [output_size]
}

impl BackwardOp for FcProjectionBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let pooled = &saved[0];
        let weight = &saved[1];

        // grad_pooled = grad_output @ weight
        let grad_pooled = grad_output.matmul_raw(weight).expect("fc grad_pooled");

        // grad_weight = grad_output^T @ pooled
        let grad_output_t = grad_output.transpose_raw(-1, -2).expect("fc grad_output transpose");
        let grad_weight = grad_output_t.matmul_raw(pooled).expect("fc grad_weight");

        // grad_bias = sum over batch of grad_output -> [output_size]
        let go_data = grad_output.to_vec();
        let batch = self.pooled_shape[0];
        let out_size = self.bias_shape[0];
        let mut grad_bias_data = vec![0.0f32; out_size];
        for b_idx in 0..batch {
            for o in 0..out_size {
                grad_bias_data[o] += go_data[b_idx * out_size + o];
            }
        }
        let grad_bias =
            Tensor::from_vec(grad_bias_data, self.bias_shape.clone()).expect("fc grad_bias");

        vec![grad_pooled, grad_weight, grad_bias]
    }

    fn name(&self) -> &str {
        "FcProjectionBackward"
    }
}

// ---------------------------------------------------------------------------
// CausalConvBlock
// ---------------------------------------------------------------------------

/// A single causal convolution block with residual connection.
///
/// Applies: output = ReLU(conv1d(input)) + residual(input)
///
/// where `residual` is either identity (if in_channels == out_channels) or a
/// learned 1x1 convolution that projects the channel dimension.
struct CausalConvBlock {
    /// Convolution weight: [out_channels, in_channels, kernel_size]
    weight: Tensor,
    /// Convolution bias: [out_channels]
    bias: Tensor,
    /// Optional 1x1 residual projection weight: [out_channels, in_channels, 1]
    residual_weight: Option<Tensor>,
    /// Optional 1x1 residual projection bias: [out_channels]
    residual_bias: Option<Tensor>,
    kernel_size: usize,
    dilation: usize,
    in_channels: usize,
    out_channels: usize,
}

impl CausalConvBlock {
    /// Create a new causal convolution block.
    ///
    /// - `in_channels`: number of input channels
    /// - `out_channels`: number of output channels
    /// - `kernel_size`: size of the convolving kernel
    /// - `dilation`: spacing between kernel elements
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, dilation: usize) -> Self {
        // Kaiming initialization scale for weight
        let fan_in = in_channels * kernel_size;
        let scale = (2.0 / fan_in as f32).sqrt();

        let mut weight = Tensor::randn([out_channels, in_channels, kernel_size]);
        weight = weight.mul_scalar_raw(scale);
        weight.set_requires_grad(true);

        let mut bias = Tensor::zeros([out_channels]);
        bias.set_requires_grad(true);

        let (residual_weight, residual_bias) = if in_channels != out_channels {
            let res_scale = (2.0 / in_channels as f32).sqrt();
            let mut rw = Tensor::randn([out_channels, in_channels, 1]);
            rw = rw.mul_scalar_raw(res_scale);
            rw.set_requires_grad(true);

            let mut rb = Tensor::zeros([out_channels]);
            rb.set_requires_grad(true);

            (Some(rw), Some(rb))
        } else {
            (None, None)
        };

        Self {
            weight,
            bias,
            residual_weight,
            residual_bias,
            kernel_size,
            dilation,
            in_channels,
            out_channels,
        }
    }

    /// Forward pass for this causal convolution block.
    ///
    /// Input shape: [batch, in_channels, seq_len]
    /// Output shape: [batch, out_channels, seq_len]
    fn forward(&self, input: &Tensor) -> SwetsResult<Tensor> {
        let shape = input.shape().to_vec();
        assert_eq!(shape.len(), 3, "CausalConvBlock: expected 3D input [batch, channels, seq_len]");
        let batch = shape[0];
        let in_ch = shape[1];
        let seq_len = shape[2];
        assert_eq!(in_ch, self.in_channels);

        let ks = self.kernel_size;
        let dil = self.dilation;
        let padding = (ks - 1) * dil;
        let padded_len = seq_len + padding;

        // ---------------------------------------------------------------
        // 1. Causal padding (left-pad with zeros)
        // ---------------------------------------------------------------
        let in_data = input.to_vec();
        let mut padded_data = vec![0.0f32; batch * in_ch * padded_len];
        for b_idx in 0..batch {
            for c in 0..in_ch {
                for t in 0..seq_len {
                    padded_data[b_idx * in_ch * padded_len + c * padded_len + padding + t] =
                        in_data[b_idx * in_ch * seq_len + c * seq_len + t];
                }
            }
        }
        let padded_input =
            Tensor::from_vec(padded_data, vec![batch, in_ch, padded_len])?;

        // ---------------------------------------------------------------
        // 2. Dilated convolution: output[b][o][t] = bias[o] + sum_i sum_k weight[o][i][k] * padded[b][i][t + k*dil]
        // ---------------------------------------------------------------
        let w_data = self.weight.to_vec();
        let b_data = self.bias.to_vec();
        let pad_data_ref = padded_input.to_vec();
        let out_ch = self.out_channels;
        let out_len = seq_len; // causal padding preserves sequence length

        let mut conv_data = vec![0.0f32; batch * out_ch * out_len];
        for b_idx in 0..batch {
            for o in 0..out_ch {
                for t in 0..out_len {
                    let mut val = b_data[o];
                    for i in 0..in_ch {
                        for k in 0..ks {
                            let p_idx = t + k * dil;
                            // p_idx is always < padded_len because:
                            //   max t = seq_len - 1, max k*dil = (ks-1)*dil = padding
                            //   t + k*dil <= seq_len - 1 + padding = padded_len - 1
                            let pv = pad_data_ref
                                [b_idx * in_ch * padded_len + i * padded_len + p_idx];
                            val += w_data[o * in_ch * ks + i * ks + k] * pv;
                        }
                    }
                    conv_data[b_idx * out_ch * out_len + o * out_len + t] = val;
                }
            }
        }
        let conv_output = Tensor::from_vec(conv_data, vec![batch, out_ch, out_len])?;

        // Record causal conv1d on tape
        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(CausalConv1dBackward {
                    input_shape: shape.clone(),
                    out_channels: out_ch,
                    kernel_size: ks,
                    dilation: dil,
                    padding,
                }),
                output_id: conv_output.id(),
                input_ids: vec![input.id(), self.weight.id(), self.bias.id()],
                saved_tensors: vec![padded_input, self.weight.clone(), self.bias.clone()],
            };
            tape::record_op(entry);
        }

        // ---------------------------------------------------------------
        // 3. ReLU activation
        // ---------------------------------------------------------------
        let pre_relu = conv_output.clone();
        let relu_data: Vec<f32> = conv_output
            .to_vec()
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();
        let activated =
            Tensor::from_vec(relu_data, vec![batch, out_ch, out_len])?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(ReLU3dBackward),
                output_id: activated.id(),
                input_ids: vec![conv_output.id()],
                saved_tensors: vec![pre_relu],
            };
            tape::record_op(entry);
        }

        // ---------------------------------------------------------------
        // 4. Residual connection
        // ---------------------------------------------------------------
        let residual = if let (Some(rw), Some(rb)) = (&self.residual_weight, &self.residual_bias) {
            // 1x1 convolution for channel projection
            let rw_data = rw.to_vec();
            let rb_data = rb.to_vec();

            let mut res_data = vec![0.0f32; batch * out_ch * seq_len];
            for b_idx in 0..batch {
                for o in 0..out_ch {
                    for t in 0..seq_len {
                        let mut val = rb_data[o];
                        for i in 0..in_ch {
                            val += rw_data[o * in_ch + i]
                                * in_data[b_idx * in_ch * seq_len + i * seq_len + t];
                        }
                        res_data[b_idx * out_ch * seq_len + o * seq_len + t] = val;
                    }
                }
            }
            let res_tensor =
                Tensor::from_vec(res_data, vec![batch, out_ch, seq_len])?;

            if tape::is_recording() {
                let entry = TapeEntry {
                    backward_op: Box::new(Residual1x1Backward {
                        input_shape: shape.clone(),
                        out_channels: out_ch,
                    }),
                    output_id: res_tensor.id(),
                    input_ids: vec![input.id(), rw.id(), rb.id()],
                    saved_tensors: vec![input.clone(), rw.clone()],
                };
                tape::record_op(entry);
            }

            res_tensor
        } else {
            // Identity residual
            input.clone()
        };

        // ---------------------------------------------------------------
        // 5. Add residual + activated
        // ---------------------------------------------------------------
        let act_data = activated.to_vec();
        let res_data = residual.to_vec();
        let summed: Vec<f32> = act_data.iter().zip(res_data.iter()).map(|(&a, &r)| a + r).collect();
        let output =
            Tensor::from_vec(summed, vec![batch, out_ch, out_len])?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(Add3dBackward),
                output_id: output.id(),
                input_ids: vec![activated.id(), residual.id()],
                saved_tensors: vec![],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    /// Collect references to all learnable parameters.
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight, &self.bias];
        if let Some(ref rw) = self.residual_weight {
            params.push(rw);
        }
        if let Some(ref rb) = self.residual_bias {
            params.push(rb);
        }
        params
    }

    /// Collect mutable references to all learnable parameters.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = vec![&mut self.weight, &mut self.bias];
        if let Some(ref mut rw) = self.residual_weight {
            params.push(rw);
        }
        if let Some(ref mut rb) = self.residual_bias {
            params.push(rb);
        }
        params
    }
}

// ---------------------------------------------------------------------------
// TCN (Temporal Convolutional Network)
// ---------------------------------------------------------------------------

/// Temporal Convolutional Network (FR-800).
///
/// Architecture:
/// - Stack of [`CausalConvBlock`]s with exponentially increasing dilation
///   (dilation = 2^i for layer i), providing an exponentially growing
///   receptive field.
/// - Each block applies causal (left-only) padding so the output sequence
///   length equals the input sequence length.
/// - Each block includes a residual connection (with a learned 1x1 projection
///   when the channel count changes).
/// - After all blocks: global average pooling over the time dimension,
///   followed by a fully-connected projection to `output_size`.
///
/// Input:  `[batch, input_size, seq_len]`
/// Output: `[batch, output_size]`
///
/// # Example
///
/// ```ignore
/// use swets::TCN;
/// use swets::Layer;
/// use swets::Tensor;
///
/// let mut tcn = TCN::new(3, 10, 16, 3, 4);
/// let input = Tensor::randn([2, 3, 50]); // batch=2, features=3, seq_len=50
/// let output = tcn.forward(&input).unwrap();
/// assert_eq!(output.shape(), &[2, 10]);
/// ```
pub struct TCN {
    blocks: Vec<CausalConvBlock>,
    /// Final projection weight: [output_size, num_channels]
    fc: Tensor,
    /// Final projection bias: [output_size]
    fc_bias: Tensor,
    num_channels: usize,
    output_size: usize,
}

impl TCN {
    /// Create a new Temporal Convolutional Network.
    ///
    /// - `input_size`: number of input channels (features per time step)
    /// - `output_size`: number of output classes or regression targets
    /// - `num_channels`: hidden channel width used in all causal blocks
    /// - `kernel_size`: temporal kernel size for dilated convolutions
    /// - `num_layers`: number of stacked causal convolution blocks
    pub fn new(
        input_size: usize,
        output_size: usize,
        num_channels: usize,
        kernel_size: usize,
        num_layers: usize,
    ) -> Self {
        assert!(num_layers > 0, "TCN requires at least one layer");
        assert!(kernel_size > 0, "kernel_size must be > 0");

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let dilation = 1 << i; // 2^i
            let in_ch = if i == 0 { input_size } else { num_channels };
            blocks.push(CausalConvBlock::new(in_ch, num_channels, kernel_size, dilation));
        }

        // Final fully-connected projection
        let fc_scale = (2.0 / num_channels as f32).sqrt();
        let mut fc = Tensor::randn([output_size, num_channels]);
        fc = fc.mul_scalar_raw(fc_scale);
        fc.set_requires_grad(true);

        let mut fc_bias = Tensor::zeros([output_size]);
        fc_bias.set_requires_grad(true);

        Self {
            blocks,
            fc,
            fc_bias,
            num_channels,
            output_size,
        }
    }

    /// Returns the number of hidden channels used by the causal blocks.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Returns the output size of the final projection.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns the number of causal convolution blocks.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Computes the receptive field of the network in time steps.
    ///
    /// For a TCN with `L` layers, kernel size `k`, and dilation `2^i`:
    ///   receptive_field = 1 + (k - 1) * sum_{i=0}^{L-1} 2^i
    ///                   = 1 + (k - 1) * (2^L - 1)
    pub fn receptive_field(&self) -> usize {
        if self.blocks.is_empty() {
            return 1;
        }
        let ks = self.blocks[0].kernel_size;
        let num_layers = self.blocks.len();
        1 + (ks - 1) * ((1 << num_layers) - 1)
    }
}

impl Layer for TCN {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let shape = input.shape().to_vec();
        assert_eq!(
            shape.len(),
            3,
            "TCN: expected 3D input [batch, input_size, seq_len], got {}D",
            shape.len()
        );

        let batch = shape[0];
        let seq_len = shape[2];

        // ---------------------------------------------------------------
        // Pass through causal conv blocks
        // ---------------------------------------------------------------
        let mut x = input.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        // x: [batch, num_channels, seq_len]

        // ---------------------------------------------------------------
        // Global average pooling over the time dimension
        // ---------------------------------------------------------------
        let x_data = x.to_vec();
        let ch = self.num_channels;
        let mut pooled_data = vec![0.0f32; batch * ch];
        let inv_len = 1.0 / seq_len as f32;

        for b_idx in 0..batch {
            for c in 0..ch {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    sum += x_data[b_idx * ch * seq_len + c * seq_len + t];
                }
                pooled_data[b_idx * ch + c] = sum * inv_len;
            }
        }
        let pooled = Tensor::from_vec(pooled_data, vec![batch, ch])?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(GlobalAvgPoolBackward {
                    input_shape: vec![batch, ch, seq_len],
                }),
                output_id: pooled.id(),
                input_ids: vec![x.id()],
                saved_tensors: vec![],
            };
            tape::record_op(entry);
        }

        // ---------------------------------------------------------------
        // Fully-connected projection: output = pooled @ fc^T + fc_bias
        // ---------------------------------------------------------------
        let fc_t = self.fc.transpose_raw(-1, -2)?;
        let matmul_result = pooled.matmul_raw(&fc_t)?;
        let output = matmul_result.add_raw(&self.fc_bias)?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(FcProjectionBackward {
                    pooled_shape: vec![batch, ch],
                    bias_shape: vec![self.output_size],
                }),
                output_id: output.id(),
                input_ids: vec![pooled.id(), self.fc.id(), self.fc_bias.id()],
                saved_tensors: vec![pooled, self.fc.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params: Vec<&Tensor> = Vec::new();
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.push(&self.fc);
        params.push(&self.fc_bias);
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = Vec::new();
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
        }
        params.push(&mut self.fc);
        params.push(&mut self.fc_bias);
        params
    }
}
