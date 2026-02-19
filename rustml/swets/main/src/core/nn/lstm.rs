use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;

/// Parameters for a single LSTM layer.
struct LSTMLayerParams {
    w_ih: Tensor, // [4*hidden_size, input_size_for_layer]
    w_hh: Tensor, // [4*hidden_size, hidden_size]
    b_ih: Tensor, // [4*hidden_size]
    b_hh: Tensor, // [4*hidden_size]
}

/// LSTM layer with multi-layer support and internal state management (FR-303).
///
/// Implements the standard LSTM equations per timestep per layer:
///   gates = W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh
///   i, f, g, o = split(gates, 4)
///   i_t = sigmoid(i), f_t = sigmoid(f), g_t = tanh(g), o_t = sigmoid(o)
///   c_t = f_t * c_{t-1} + i_t * g_t
///   h_t = o_t * tanh(c_t)
///
/// Input shape:  [batch, seq_len, input_size]
/// Output shape: [batch, seq_len, hidden_size]
///
/// State is managed internally per design risk mitigation (SRS 5.5).
/// Call `reset_state()` between sequences to clear hidden/cell states.
pub struct LSTM {
    layers: Vec<LSTMLayerParams>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    /// Hidden state: conceptually [num_layers, batch, hidden_size], stored as Vec of per-layer tensors.
    h: Option<Vec<Tensor>>,
    /// Cell state: conceptually [num_layers, batch, hidden_size], stored as Vec of per-layer tensors.
    c: Option<Vec<Tensor>>,
}

impl LSTM {
    /// Creates a new LSTM with Xavier-initialized weights and zero biases.
    ///
    /// - `input_size`: dimensionality of the input features
    /// - `hidden_size`: number of hidden units per layer
    /// - `num_layers`: number of stacked LSTM layers
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        assert!(num_layers >= 1, "LSTM must have at least 1 layer");
        assert!(hidden_size > 0, "hidden_size must be positive");
        assert!(input_size > 0, "input_size must be positive");

        let mut layers = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let layer_input_size = if layer_idx == 0 { input_size } else { hidden_size };
            let gate_size = 4 * hidden_size;

            // Xavier uniform initialization: scale = sqrt(6 / (fan_in + fan_out))
            let scale_ih = (6.0 / (layer_input_size + gate_size) as f32).sqrt();
            let mut w_ih = Tensor::randn([gate_size, layer_input_size]);
            w_ih = w_ih.mul_scalar_raw(scale_ih);
            w_ih.set_requires_grad(true);

            let scale_hh = (6.0 / (hidden_size + gate_size) as f32).sqrt();
            let mut w_hh = Tensor::randn([gate_size, hidden_size]);
            w_hh = w_hh.mul_scalar_raw(scale_hh);
            w_hh.set_requires_grad(true);

            let mut b_ih = Tensor::zeros([gate_size]);
            b_ih.set_requires_grad(true);

            let mut b_hh = Tensor::zeros([gate_size]);
            b_hh.set_requires_grad(true);

            layers.push(LSTMLayerParams { w_ih, w_hh, b_ih, b_hh });
        }

        Self {
            layers,
            input_size,
            hidden_size,
            num_layers,
            h: None,
            c: None,
        }
    }

    /// Clears internal hidden and cell states.
    /// Call this between independent sequences.
    pub fn reset_state(&mut self) {
        self.h = None;
        self.c = None;
    }

    /// Returns the input size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

impl Layer for LSTM {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let shape = input.shape().to_vec();
        assert_eq!(
            shape.len(),
            3,
            "LSTM input must be [batch, seq_len, input_size], got {:?}",
            shape
        );
        let batch = shape[0];
        let seq_len = shape[1];
        let in_size = shape[2];
        assert_eq!(
            in_size, self.input_size,
            "LSTM input_size mismatch: expected {}, got {}",
            self.input_size, in_size
        );

        let hidden = self.hidden_size;

        // Initialize states if needed (zeros for each layer)
        let mut h_states: Vec<Tensor> = match &self.h {
            Some(h) => h.clone(),
            None => (0..self.num_layers)
                .map(|_| Tensor::zeros(vec![batch, hidden]))
                .collect(),
        };
        let mut c_states: Vec<Tensor> = match &self.c {
            Some(c) => c.clone(),
            None => (0..self.num_layers)
                .map(|_| Tensor::zeros(vec![batch, hidden]))
                .collect(),
        };

        let input_data = input.to_vec();

        // We will collect all intermediates for the backward pass.
        // Per timestep, per layer: (i_gate, f_gate, g_gate, o_gate, c_new, tanh_c_new)
        // Plus the input to each layer at each timestep.
        // Also store the h and c at each step for each layer.
        let mut all_gate_i: Vec<Vec<Vec<f32>>> = vec![vec![]; self.num_layers]; // [layer][timestep] -> flat data
        let mut all_gate_f: Vec<Vec<Vec<f32>>> = vec![vec![]; self.num_layers];
        let mut all_gate_g: Vec<Vec<Vec<f32>>> = vec![vec![]; self.num_layers];
        let mut all_gate_o: Vec<Vec<Vec<f32>>> = vec![vec![]; self.num_layers];
        let mut all_tanh_c: Vec<Vec<Vec<f32>>> = vec![vec![]; self.num_layers]; // tanh(c_new)
        let mut all_h_prev: Vec<Vec<Vec<f32>>> = vec![vec![]; self.num_layers]; // h before this timestep
        let mut all_c_prev: Vec<Vec<Vec<f32>>> = vec![vec![]; self.num_layers]; // c before this timestep
        let mut all_layer_inputs: Vec<Vec<Vec<f32>>> = vec![vec![]; self.num_layers]; // input to each layer per timestep

        // Output: [batch, seq_len, hidden_size]
        let mut output_data = vec![0.0f32; batch * seq_len * hidden];

        // Preload weight/bias data for performance (avoid repeated to_vec calls)
        struct LayerWeights {
            w_ih: Vec<f32>,
            w_hh: Vec<f32>,
            b_ih: Vec<f32>,
            b_hh: Vec<f32>,
            layer_input_size: usize,
        }

        let layer_weights: Vec<LayerWeights> = (0..self.num_layers)
            .map(|l| {
                let lis = if l == 0 { self.input_size } else { hidden };
                LayerWeights {
                    w_ih: self.layers[l].w_ih.to_vec(),
                    w_hh: self.layers[l].w_hh.to_vec(),
                    b_ih: self.layers[l].b_ih.to_vec(),
                    b_hh: self.layers[l].b_hh.to_vec(),
                    layer_input_size: lis,
                }
            })
            .collect();

        // Current h/c state data per layer: [batch * hidden]
        let mut h_data: Vec<Vec<f32>> = h_states.iter().map(|t| t.to_vec()).collect();
        let mut c_data: Vec<Vec<f32>> = c_states.iter().map(|t| t.to_vec()).collect();

        for t in 0..seq_len {
            // Extract this timestep's input from the flattened input_data: [batch, input_size]
            let mut timestep_input = vec![0.0f32; batch * self.input_size];
            for b in 0..batch {
                let src_start = b * seq_len * in_size + t * in_size;
                let dst_start = b * in_size;
                timestep_input[dst_start..dst_start + in_size]
                    .copy_from_slice(&input_data[src_start..src_start + in_size]);
            }

            // The input flowing through layers for this timestep
            let mut layer_input = timestep_input;

            for l in 0..self.num_layers {
                let lw = &layer_weights[l];
                let lis = lw.layer_input_size;
                let gate_size = 4 * hidden;

                // Save input and previous states for backward
                all_layer_inputs[l].push(layer_input.clone());
                all_h_prev[l].push(h_data[l].clone());
                all_c_prev[l].push(c_data[l].clone());

                // Compute gates for each batch element
                // gates = W_ih @ x + b_ih + W_hh @ h + b_hh
                // W_ih: [gate_size, lis], x: [lis] per batch
                // W_hh: [gate_size, hidden], h: [hidden] per batch
                let mut gates = vec![0.0f32; batch * gate_size];

                for b in 0..batch {
                    let x_start = b * lis;
                    let h_start = b * hidden;
                    let g_start = b * gate_size;

                    for g in 0..gate_size {
                        let mut val = lw.b_ih[g] + lw.b_hh[g];

                        // W_ih[g, :] @ x
                        let w_ih_row_start = g * lis;
                        for k in 0..lis {
                            val += lw.w_ih[w_ih_row_start + k] * layer_input[x_start + k];
                        }

                        // W_hh[g, :] @ h
                        let w_hh_row_start = g * hidden;
                        for k in 0..hidden {
                            val += lw.w_hh[w_hh_row_start + k] * h_data[l][h_start + k];
                        }

                        gates[g_start + g] = val;
                    }
                }

                // Split gates and apply activations
                let mut i_gate = vec![0.0f32; batch * hidden];
                let mut f_gate = vec![0.0f32; batch * hidden];
                let mut g_gate = vec![0.0f32; batch * hidden];
                let mut o_gate = vec![0.0f32; batch * hidden];

                for b in 0..batch {
                    let g_start = b * gate_size;
                    let h_start = b * hidden;
                    for h_idx in 0..hidden {
                        // Gates order: i, f, g, o (each hidden-sized)
                        let i_raw = gates[g_start + h_idx];
                        let f_raw = gates[g_start + hidden + h_idx];
                        let g_raw = gates[g_start + 2 * hidden + h_idx];
                        let o_raw = gates[g_start + 3 * hidden + h_idx];

                        i_gate[h_start + h_idx] = sigmoid(i_raw);
                        f_gate[h_start + h_idx] = sigmoid(f_raw);
                        g_gate[h_start + h_idx] = g_raw.tanh();
                        o_gate[h_start + h_idx] = sigmoid(o_raw);
                    }
                }

                // c_new = f * c_old + i * g
                // h_new = o * tanh(c_new)
                let mut c_new = vec![0.0f32; batch * hidden];
                let mut tanh_c_new = vec![0.0f32; batch * hidden];
                let mut h_new = vec![0.0f32; batch * hidden];

                for idx in 0..batch * hidden {
                    c_new[idx] = f_gate[idx] * c_data[l][idx] + i_gate[idx] * g_gate[idx];
                    tanh_c_new[idx] = c_new[idx].tanh();
                    h_new[idx] = o_gate[idx] * tanh_c_new[idx];
                }

                // Save intermediates for backward
                all_gate_i[l].push(i_gate);
                all_gate_f[l].push(f_gate);
                all_gate_g[l].push(g_gate);
                all_gate_o[l].push(o_gate);
                all_tanh_c[l].push(tanh_c_new);

                // Update state
                h_data[l] = h_new.clone();
                c_data[l] = c_new;

                // Output of this layer becomes input to the next layer
                layer_input = h_new;
            }

            // Write the final layer's h to the output
            let final_h = &h_data[self.num_layers - 1];
            for b in 0..batch {
                let dst_start = b * seq_len * hidden + t * hidden;
                let src_start = b * hidden;
                output_data[dst_start..dst_start + hidden]
                    .copy_from_slice(&final_h[src_start..src_start + hidden]);
            }
        }

        // Update internal states as Tensors
        for l in 0..self.num_layers {
            h_states[l] = Tensor::from_vec(h_data[l].clone(), vec![batch, hidden])?;
            c_states[l] = Tensor::from_vec(c_data[l].clone(), vec![batch, hidden])?;
        }
        self.h = Some(h_states);
        self.c = Some(c_states);

        let output = Tensor::from_vec(output_data, vec![batch, seq_len, hidden])?;

        // Record on tape for backward pass
        if tape::is_recording() {
            // Collect all weight/bias tensor IDs for input_ids
            let mut input_ids = vec![input.id()];
            let mut saved_tensors = vec![input.clone()];

            for l in 0..self.num_layers {
                input_ids.push(self.layers[l].w_ih.id());
                input_ids.push(self.layers[l].w_hh.id());
                input_ids.push(self.layers[l].b_ih.id());
                input_ids.push(self.layers[l].b_hh.id());

                saved_tensors.push(self.layers[l].w_ih.clone());
                saved_tensors.push(self.layers[l].w_hh.clone());
                saved_tensors.push(self.layers[l].b_ih.clone());
                saved_tensors.push(self.layers[l].b_hh.clone());
            }

            let entry = TapeEntry {
                backward_op: Box::new(LSTMBackward {
                    input_size: self.input_size,
                    hidden_size: hidden,
                    num_layers: self.num_layers,
                    batch,
                    seq_len,
                    gate_i: all_gate_i,
                    gate_f: all_gate_f,
                    gate_g: all_gate_g,
                    gate_o: all_gate_o,
                    tanh_c: all_tanh_c,
                    h_prev: all_h_prev,
                    c_prev: all_c_prev,
                    layer_inputs: all_layer_inputs,
                }),
                output_id: output.id(),
                input_ids,
                saved_tensors,
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for layer in &self.layers {
            params.push(&layer.w_ih);
            params.push(&layer.w_hh);
            params.push(&layer.b_ih);
            params.push(&layer.b_hh);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for layer in &mut self.layers {
            params.push(&mut layer.w_ih);
            params.push(&mut layer.w_hh);
            params.push(&mut layer.b_ih);
            params.push(&mut layer.b_hh);
        }
        params
    }
}

/// Inline sigmoid for scalar values.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Backward pass for the LSTM using BPTT (Backpropagation Through Time).
///
/// This stores all gate activations and states needed to compute gradients.
/// Gradients flow backward through timesteps and layers.
///
/// For each TapeEntry input_ids ordering:
///   [0] = input tensor
///   [1 + l*4 + 0] = w_ih for layer l
///   [1 + l*4 + 1] = w_hh for layer l
///   [1 + l*4 + 2] = b_ih for layer l
///   [1 + l*4 + 3] = b_hh for layer l
struct LSTMBackward {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    batch: usize,
    seq_len: usize,
    // Per layer, per timestep: flattened [batch * hidden]
    gate_i: Vec<Vec<Vec<f32>>>,
    gate_f: Vec<Vec<Vec<f32>>>,
    gate_g: Vec<Vec<Vec<f32>>>,
    gate_o: Vec<Vec<Vec<f32>>>,
    tanh_c: Vec<Vec<Vec<f32>>>,     // tanh(c_new)
    h_prev: Vec<Vec<Vec<f32>>>,     // h before update
    c_prev: Vec<Vec<Vec<f32>>>,     // c before update
    layer_inputs: Vec<Vec<Vec<f32>>>, // input to each layer per timestep
}

impl BackwardOp for LSTMBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let hidden = self.hidden_size;
        let batch = self.batch;
        let seq_len = self.seq_len;
        let num_layers = self.num_layers;

        let grad_output_data = grad_output.to_vec();
        // grad_output shape: [batch, seq_len, hidden]

        // Recover weight data from saved tensors
        // saved[0] = input, saved[1 + l*4 + k] = weight/bias for layer l
        struct LayerWeightData {
            w_ih: Vec<f32>,
            w_hh: Vec<f32>,
            layer_input_size: usize,
        }

        let layer_wd: Vec<LayerWeightData> = (0..num_layers)
            .map(|l| {
                let lis = if l == 0 { self.input_size } else { hidden };
                LayerWeightData {
                    w_ih: saved[1 + l * 4].to_vec(),
                    w_hh: saved[1 + l * 4 + 1].to_vec(),
                    layer_input_size: lis,
                }
            })
            .collect();

        // Allocate gradient accumulators for weights and biases
        let mut grad_w_ih: Vec<Vec<f32>> = (0..num_layers)
            .map(|l| {
                let lis = layer_wd[l].layer_input_size;
                vec![0.0f32; 4 * hidden * lis]
            })
            .collect();
        let mut grad_w_hh: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; 4 * hidden * hidden])
            .collect();
        let mut grad_b_ih: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; 4 * hidden])
            .collect();
        let mut grad_b_hh: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0.0f32; 4 * hidden])
            .collect();

        // Gradient of the input: [batch, seq_len, input_size]
        let mut grad_input_data = vec![0.0f32; batch * seq_len * self.input_size];

        // For each layer, we need dh_next and dc_next to propagate backward through time
        // The grad flowing into the top layer at each timestep comes from grad_output
        // For lower layers, grad comes from the layer above via the weight matrices

        // We process layers from top to bottom, timesteps from last to first.
        // For the top layer (num_layers - 1): d_h comes from grad_output at each timestep.
        // For lower layers: d_h comes from the "grad w.r.t. input" of the layer above.

        // We store per-layer, per-timestep "grad w.r.t. layer input" to pass down.
        // grad_layer_input[l][t] = gradient of loss w.r.t. input of layer l at timestep t
        // For the top layer, this is grad_output[:, t, :].
        // After computing BPTT for layer l, we get grad w.r.t. x_t for that layer,
        // which becomes grad_layer_input[l-1][t] (the grad flowing into layer l-1's output = h).

        // Start from the top layer and work down
        // grad_from_above[t] = [batch * hidden] gradient flowing into layer l at timestep t
        let mut grad_from_above: Vec<Vec<f32>> = (0..seq_len)
            .map(|t| {
                let mut v = vec![0.0f32; batch * hidden];
                for b in 0..batch {
                    let src = b * seq_len * hidden + t * hidden;
                    let dst = b * hidden;
                    v[dst..dst + hidden].copy_from_slice(&grad_output_data[src..src + hidden]);
                }
                v
            })
            .collect();

        for l in (0..num_layers).rev() {
            let lis = layer_wd[l].layer_input_size;

            // BPTT for this layer
            let mut dh_next = vec![0.0f32; batch * hidden];
            let mut dc_next = vec![0.0f32; batch * hidden];

            // Gradient w.r.t. the input of this layer at each timestep
            let mut grad_x_per_t: Vec<Vec<f32>> = vec![vec![]; seq_len];

            for t in (0..seq_len).rev() {
                let i_gate = &self.gate_i[l][t];
                let f_gate = &self.gate_f[l][t];
                let g_gate = &self.gate_g[l][t];
                let o_gate = &self.gate_o[l][t];
                let tanh_c = &self.tanh_c[l][t];
                let c_prev = &self.c_prev[l][t];
                let h_prev = &self.h_prev[l][t];
                let x_t = &self.layer_inputs[l][t];

                // Total gradient on h_t = grad from above + grad from next timestep
                let mut dh = vec![0.0f32; batch * hidden];
                for idx in 0..batch * hidden {
                    dh[idx] = grad_from_above[t][idx] + dh_next[idx];
                }

                // d_tanh_c = dh * o_gate
                // dc = d_tanh_c * (1 - tanh_c^2) + dc_next
                //    + dc from f_gate of next timestep (already in dc_next)
                let mut dc = vec![0.0f32; batch * hidden];
                for idx in 0..batch * hidden {
                    let d_tanh_c = dh[idx] * o_gate[idx];
                    dc[idx] = d_tanh_c * (1.0 - tanh_c[idx] * tanh_c[idx]) + dc_next[idx];
                }

                // Gate gradients (pre-activation)
                // di_raw = dc * g_gate * i_gate * (1 - i_gate)   [sigmoid derivative]
                // df_raw = dc * c_prev * f_gate * (1 - f_gate)   [sigmoid derivative]
                // dg_raw = dc * i_gate * (1 - g_gate^2)          [tanh derivative]
                // do_raw = dh * tanh_c * o_gate * (1 - o_gate)   [sigmoid derivative]
                let mut d_gates = vec![0.0f32; batch * 4 * hidden];
                for b in 0..batch {
                    let h_off = b * hidden;
                    let g_off = b * 4 * hidden;
                    for h_idx in 0..hidden {
                        let idx = h_off + h_idx;

                        let di_raw = dc[idx] * g_gate[idx] * i_gate[idx] * (1.0 - i_gate[idx]);
                        let df_raw =
                            dc[idx] * c_prev[idx] * f_gate[idx] * (1.0 - f_gate[idx]);
                        let dg_raw = dc[idx] * i_gate[idx] * (1.0 - g_gate[idx] * g_gate[idx]);
                        let do_raw =
                            dh[idx] * tanh_c[idx] * o_gate[idx] * (1.0 - o_gate[idx]);

                        d_gates[g_off + h_idx] = di_raw;
                        d_gates[g_off + hidden + h_idx] = df_raw;
                        d_gates[g_off + 2 * hidden + h_idx] = dg_raw;
                        d_gates[g_off + 3 * hidden + h_idx] = do_raw;
                    }
                }

                // Accumulate weight gradients:
                // grad_w_ih += d_gates^T @ x_t   (per batch element, accumulated)
                // grad_w_hh += d_gates^T @ h_prev (per batch element, accumulated)
                // grad_b_ih += sum over batch of d_gates
                // grad_b_hh += sum over batch of d_gates
                let gate_size = 4 * hidden;

                for b in 0..batch {
                    let g_off = b * gate_size;
                    let x_off = b * lis;
                    let h_off = b * hidden;

                    for g in 0..gate_size {
                        let dg = d_gates[g_off + g];
                        grad_b_ih[l][g] += dg;
                        grad_b_hh[l][g] += dg;

                        // w_ih gradient: outer product d_gates[g] * x[k]
                        let w_row = g * lis;
                        for k in 0..lis {
                            grad_w_ih[l][w_row + k] += dg * x_t[x_off + k];
                        }

                        // w_hh gradient: outer product d_gates[g] * h_prev[k]
                        let w_row_hh = g * hidden;
                        for k in 0..hidden {
                            grad_w_hh[l][w_row_hh + k] += dg * h_prev[h_off + k];
                        }
                    }
                }

                // Compute dh_next = W_hh^T @ d_gates (for next timestep backward)
                // dh_next[b, k] = sum_g W_hh[g, k] * d_gates[b, g]
                dh_next = vec![0.0f32; batch * hidden];
                for b in 0..batch {
                    let g_off = b * gate_size;
                    let h_off = b * hidden;
                    for k in 0..hidden {
                        let mut val = 0.0f32;
                        for g in 0..gate_size {
                            val += layer_wd[l].w_hh[g * hidden + k] * d_gates[g_off + g];
                        }
                        dh_next[h_off + k] = val;
                    }
                }

                // dc_next = dc * f_gate (gradient flows through the forget gate)
                dc_next = vec![0.0f32; batch * hidden];
                for idx in 0..batch * hidden {
                    dc_next[idx] = dc[idx] * f_gate[idx];
                }

                // Compute grad w.r.t. x_t = W_ih^T @ d_gates
                // dx[b, k] = sum_g W_ih[g, k] * d_gates[b, g]
                let mut dx = vec![0.0f32; batch * lis];
                for b in 0..batch {
                    let g_off = b * gate_size;
                    let x_off = b * lis;
                    for k in 0..lis {
                        let mut val = 0.0f32;
                        for g in 0..gate_size {
                            val += layer_wd[l].w_ih[g * lis + k] * d_gates[g_off + g];
                        }
                        dx[x_off + k] = val;
                    }
                }

                grad_x_per_t[t] = dx;
            }

            if l == 0 {
                // grad_x_per_t flows into grad_input_data
                for t in 0..seq_len {
                    let dx = &grad_x_per_t[t];
                    for b in 0..batch {
                        let dst = b * seq_len * self.input_size + t * self.input_size;
                        let src = b * self.input_size;
                        for k in 0..self.input_size {
                            grad_input_data[dst + k] += dx[src + k];
                        }
                    }
                }
            } else {
                // grad_x_per_t becomes grad_from_above for the layer below
                grad_from_above = grad_x_per_t;
            }
        }

        // Build result tensors in the same order as input_ids:
        // [0] = grad_input
        // [1 + l*4 + 0] = grad_w_ih for layer l
        // [1 + l*4 + 1] = grad_w_hh for layer l
        // [1 + l*4 + 2] = grad_b_ih for layer l
        // [1 + l*4 + 3] = grad_b_hh for layer l
        let mut grads = Vec::with_capacity(1 + num_layers * 4);

        grads.push(
            Tensor::from_vec(grad_input_data, vec![batch, seq_len, self.input_size])
                .expect("lstm grad_input"),
        );

        for l in 0..num_layers {
            let lis = layer_wd[l].layer_input_size;
            grads.push(
                Tensor::from_vec(grad_w_ih[l].clone(), vec![4 * hidden, lis])
                    .expect("lstm grad_w_ih"),
            );
            grads.push(
                Tensor::from_vec(grad_w_hh[l].clone(), vec![4 * hidden, hidden])
                    .expect("lstm grad_w_hh"),
            );
            grads.push(
                Tensor::from_vec(grad_b_ih[l].clone(), vec![4 * hidden])
                    .expect("lstm grad_b_ih"),
            );
            grads.push(
                Tensor::from_vec(grad_b_hh[l].clone(), vec![4 * hidden])
                    .expect("lstm grad_b_hh"),
            );
        }

        grads
    }

    fn name(&self) -> &str {
        "LSTMBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_output_shape() {
        let mut lstm = LSTM::new(10, 20, 2);
        let input = Tensor::randn([4, 5, 10]); // batch=4, seq_len=5, input_size=10
        let output = lstm.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 5, 20]);
    }

    #[test]
    fn test_lstm_single_layer() {
        let mut lstm = LSTM::new(3, 4, 1);
        let input = Tensor::randn([2, 3, 3]); // batch=2, seq_len=3, input_size=3
        let output = lstm.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);

        // Output values should be bounded by tanh (through o_gate * tanh(c))
        let data = output.to_vec();
        for &v in &data {
            assert!(v >= -1.0 && v <= 1.0, "LSTM output should be in [-1, 1], got {}", v);
        }
    }

    #[test]
    fn test_lstm_state_persistence() {
        let mut lstm = LSTM::new(3, 4, 1);
        let input = Tensor::randn([1, 2, 3]);

        let out1 = lstm.forward(&input).unwrap();
        let out2 = lstm.forward(&input).unwrap(); // state carried over

        // Outputs should differ because internal state changed
        let d1 = out1.to_vec();
        let d2 = out2.to_vec();
        let differs = d1.iter().zip(d2.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(differs, "Outputs should differ when state is carried over");
    }

    #[test]
    fn test_lstm_reset_state() {
        let mut lstm = LSTM::new(3, 4, 1);
        let input = Tensor::randn([1, 2, 3]);

        let out1 = lstm.forward(&input).unwrap();
        lstm.reset_state();
        let out2 = lstm.forward(&input).unwrap();

        // After reset, outputs should be identical (same initial zero state)
        let d1 = out1.to_vec();
        let d2 = out2.to_vec();
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "After reset, outputs should match: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_lstm_parameters_count() {
        let lstm = LSTM::new(10, 20, 2);
        let params = lstm.parameters();
        // 2 layers * 4 params each = 8
        assert_eq!(params.len(), 8);

        // Layer 0: w_ih [80, 10] + w_hh [80, 20] + b_ih [80] + b_hh [80]
        // = 800 + 1600 + 80 + 80 = 2560
        // Layer 1: w_ih [80, 20] + w_hh [80, 20] + b_ih [80] + b_hh [80]
        // = 1600 + 1600 + 80 + 80 = 3360
        let total: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total, 2560 + 3360);
    }
}
