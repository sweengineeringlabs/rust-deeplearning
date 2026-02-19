use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;
use crate::core::nn::activations::ReLU;
use crate::core::nn::linear::Linear;
use crate::core::ops::add::unbroadcast;

// ---------------------------------------------------------------------------
// Tape-recorded Add / Sub helpers
// ---------------------------------------------------------------------------

/// Element-wise addition recorded on the gradient tape.
fn tape_add(a: &Tensor, b: &Tensor) -> SwetsResult<Tensor> {
    let output = a.add_raw(b)?;
    if tape::is_recording() {
        tape::record_op(TapeEntry {
            backward_op: Box::new(AddBackward {
                a_shape: a.shape().to_vec(),
                b_shape: b.shape().to_vec(),
            }),
            output_id: output.id(),
            input_ids: vec![a.id(), b.id()],
            saved_tensors: vec![],
        });
    }
    Ok(output)
}

/// Element-wise subtraction recorded on the gradient tape.
fn tape_sub(a: &Tensor, b: &Tensor) -> SwetsResult<Tensor> {
    let output = a.sub_raw(b)?;
    if tape::is_recording() {
        tape::record_op(TapeEntry {
            backward_op: Box::new(SubBackward {
                a_shape: a.shape().to_vec(),
                b_shape: b.shape().to_vec(),
            }),
            output_id: output.id(),
            input_ids: vec![a.id(), b.id()],
            saved_tensors: vec![],
        });
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Backward ops for the tape helpers
// ---------------------------------------------------------------------------

/// Backward for C = A + B (with broadcasting support).
struct AddBackward {
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl BackwardOp for AddBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        let grad_a = unbroadcast(grad_output, &self.a_shape);
        let grad_b = unbroadcast(grad_output, &self.b_shape);
        vec![grad_a, grad_b]
    }

    fn name(&self) -> &str {
        "NBeatsAddBackward"
    }
}

/// Backward for C = A - B (with broadcasting support).
/// grad_A = unbroadcast(grad_output, shape_A)
/// grad_B = unbroadcast(-grad_output, shape_B)
struct SubBackward {
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl BackwardOp for SubBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        let grad_a = unbroadcast(grad_output, &self.a_shape);
        let neg_grad = grad_output.neg_raw();
        let grad_b = unbroadcast(&neg_grad, &self.b_shape);
        vec![grad_a, grad_b]
    }

    fn name(&self) -> &str {
        "NBeatsSubBackward"
    }
}

// ---------------------------------------------------------------------------
// NBeatsBlock
// ---------------------------------------------------------------------------

/// A single N-BEATS block with fully-connected layers, backcast and forecast
/// output heads.
///
/// Architecture:
///   input -> FC1 -> ReLU -> FC2 -> ReLU -> ... -> FCn -> ReLU
///                                                   |-> backcast_fc -> backcast
///                                                   |-> forecast_fc -> forecast
struct NBeatsBlock {
    fc_layers: Vec<Linear>,
    relu_activations: Vec<ReLU>,
    backcast_fc: Linear,
    forecast_fc: Linear,
    #[allow(dead_code)]
    hidden_size: usize,
}

impl NBeatsBlock {
    /// Create a new N-BEATS block.
    ///
    /// # Arguments
    /// * `input_size`     - Dimensionality of the block input (first block gets
    ///                      `backcast_size`, subsequent blocks also get `backcast_size`
    ///                      because the residual maintains that shape).
    /// * `hidden_size`    - Width of each hidden FC layer.
    /// * `num_fc_layers`  - Number of FC+ReLU layers in the shared trunk.
    /// * `backcast_size`  - Output size of the backcast head (= input window length).
    /// * `forecast_size`  - Output size of the forecast head (= prediction horizon).
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_fc_layers: usize,
        backcast_size: usize,
        forecast_size: usize,
    ) -> Self {
        assert!(num_fc_layers >= 1, "NBeatsBlock requires at least 1 FC layer");

        let mut fc_layers = Vec::with_capacity(num_fc_layers);
        let mut relu_activations = Vec::with_capacity(num_fc_layers);

        // First FC layer: input_size -> hidden_size
        fc_layers.push(Linear::new(input_size, hidden_size));
        relu_activations.push(ReLU::new());

        // Remaining FC layers: hidden_size -> hidden_size
        for _ in 1..num_fc_layers {
            fc_layers.push(Linear::new(hidden_size, hidden_size));
            relu_activations.push(ReLU::new());
        }

        let backcast_fc = Linear::new(hidden_size, backcast_size);
        let forecast_fc = Linear::new(hidden_size, forecast_size);

        Self {
            fc_layers,
            relu_activations,
            backcast_fc,
            forecast_fc,
            hidden_size,
        }
    }

    /// Forward pass through the block.
    ///
    /// Returns `(backcast, forecast)` where:
    /// - `backcast` has shape `[batch, backcast_size]`
    /// - `forecast` has shape `[batch, forecast_size]`
    fn forward(&mut self, input: &Tensor) -> SwetsResult<(Tensor, Tensor)> {
        let mut h = input.clone();

        // Pass through FC layers with ReLU activation
        for (fc, relu) in self.fc_layers.iter_mut().zip(self.relu_activations.iter_mut()) {
            h = fc.forward(&h)?;
            h = relu.forward(&h)?;
        }

        // Project to backcast and forecast
        let backcast = self.backcast_fc.forward(&h)?;
        let forecast = self.forecast_fc.forward(&h)?;

        Ok((backcast, forecast))
    }

    /// Collect all learnable parameters (immutable references).
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for fc in &self.fc_layers {
            params.extend(fc.parameters());
        }
        params.extend(self.backcast_fc.parameters());
        params.extend(self.forecast_fc.parameters());
        params
    }

    /// Collect all learnable parameters (mutable references).
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for fc in &mut self.fc_layers {
            params.extend(fc.parameters_mut());
        }
        params.extend(self.backcast_fc.parameters_mut());
        params.extend(self.forecast_fc.parameters_mut());
        params
    }
}

// ---------------------------------------------------------------------------
// NBeatsStack
// ---------------------------------------------------------------------------

/// A stack of N-BEATS blocks that share the same architecture.
struct NBeatsStack {
    blocks: Vec<NBeatsBlock>,
}

impl NBeatsStack {
    /// Create a new stack of identical N-BEATS blocks.
    fn new(
        num_blocks: usize,
        input_size: usize,
        hidden_size: usize,
        num_fc_layers: usize,
        backcast_size: usize,
        forecast_size: usize,
    ) -> Self {
        let blocks = (0..num_blocks)
            .map(|_| NBeatsBlock::new(input_size, hidden_size, num_fc_layers, backcast_size, forecast_size))
            .collect();

        Self { blocks }
    }

    /// Collect all learnable parameters (immutable references).
    fn parameters(&self) -> Vec<&Tensor> {
        self.blocks.iter().flat_map(|b| b.parameters()).collect()
    }

    /// Collect all learnable parameters (mutable references).
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.blocks.iter_mut().flat_map(|b| b.parameters_mut()).collect()
    }
}

// ---------------------------------------------------------------------------
// NBeats (public model)
// ---------------------------------------------------------------------------

/// N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series
/// Forecasting) model.
///
/// Implements the doubly-residual architecture:
/// - **Backcast subtraction**: each block's backcast is subtracted from the
///   residual signal, letting subsequent blocks focus on what earlier blocks
///   could not explain.
/// - **Forecast accumulation**: each block's forecast contribution is summed
///   into the final prediction.
///
/// # Architecture
///
/// ```text
/// input ──> [Stack 1] ──> [Stack 2] ──> ... ──> [Stack S]
///               |              |                      |
///               v              v                      v
///          stack_fcst_1 + stack_fcst_2 + ... + stack_fcst_S = output
/// ```
///
/// Within each stack, blocks are chained via the residual:
///
/// ```text
/// residual ──> Block 1 ──> residual' ──> Block 2 ──> ...
///                 |                          |
///            backcast_1                 backcast_2
///            forecast_1                 forecast_2
///               |                          |
///               +--- stack_forecast -------+--- ...
/// ```
pub struct NBeats {
    stacks: Vec<NBeatsStack>,
    backcast_size: usize,
    forecast_size: usize,
}

impl NBeats {
    /// Create a new N-BEATS model.
    ///
    /// # Arguments
    /// * `backcast_size`  - Length of the input window (lookback period).
    /// * `forecast_size`  - Length of the prediction horizon.
    /// * `num_stacks`     - Number of stacks in the model.
    /// * `num_blocks`     - Number of blocks per stack.
    /// * `hidden_size`    - Width of the hidden FC layers in each block.
    /// * `num_fc_layers`  - Number of FC+ReLU layers in each block's trunk.
    pub fn new(
        backcast_size: usize,
        forecast_size: usize,
        num_stacks: usize,
        num_blocks: usize,
        hidden_size: usize,
        num_fc_layers: usize,
    ) -> Self {
        let stacks = (0..num_stacks)
            .map(|_| {
                NBeatsStack::new(
                    num_blocks,
                    backcast_size,
                    hidden_size,
                    num_fc_layers,
                    backcast_size,
                    forecast_size,
                )
            })
            .collect();

        Self {
            stacks,
            backcast_size,
            forecast_size,
        }
    }

    /// Return the expected input size (lookback window length).
    pub fn backcast_size(&self) -> usize {
        self.backcast_size
    }

    /// Return the forecast horizon length.
    pub fn forecast_size(&self) -> usize {
        self.forecast_size
    }
}

impl Layer for NBeats {
    /// Forward pass through the full N-BEATS model.
    ///
    /// # Input
    /// `input` has shape `[batch, backcast_size]` -- a flattened window of past
    /// observations.
    ///
    /// # Output
    /// Tensor of shape `[batch, forecast_size]` -- the predicted future values.
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let batch_size = input.shape()[0];

        // Running residual that blocks progressively explain away.
        let mut residual = input.clone();

        // Accumulated forecast across all stacks.
        let mut total_forecast = Tensor::zeros([batch_size, self.forecast_size]);

        for stack in &mut self.stacks {
            // Each stack accumulates its own forecast from its blocks.
            let mut stack_forecast = Tensor::zeros([batch_size, self.forecast_size]);

            for block in &mut stack.blocks {
                let (backcast, forecast) = block.forward(&residual)?;

                // Doubly-residual: subtract backcast from residual
                residual = tape_sub(&residual, &backcast)?;

                // Accumulate this block's forecast contribution
                stack_forecast = tape_add(&stack_forecast, &forecast)?;
            }

            // Add this stack's total forecast to the global forecast
            total_forecast = tape_add(&total_forecast, &stack_forecast)?;
        }

        Ok(total_forecast)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.stacks.iter().flat_map(|s| s.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.stacks.iter_mut().flat_map(|s| s.parameters_mut()).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nbeats_output_shape() {
        let backcast_size = 10;
        let forecast_size = 5;
        let batch_size = 4;

        let mut model = NBeats::new(
            backcast_size,
            forecast_size,
            /* num_stacks */ 2,
            /* num_blocks */ 3,
            /* hidden_size */ 16,
            /* num_fc_layers */ 2,
        );

        let input = Tensor::randn([batch_size, backcast_size]);
        let output = model.forward(&input).expect("forward pass");

        assert_eq!(output.shape(), &[batch_size, forecast_size]);
    }

    #[test]
    fn test_nbeats_single_stack_single_block() {
        let backcast_size = 8;
        let forecast_size = 3;
        let batch_size = 2;

        let mut model = NBeats::new(
            backcast_size,
            forecast_size,
            /* num_stacks */ 1,
            /* num_blocks */ 1,
            /* hidden_size */ 8,
            /* num_fc_layers */ 1,
        );

        let input = Tensor::randn([batch_size, backcast_size]);
        let output = model.forward(&input).expect("forward pass");

        assert_eq!(output.shape(), &[batch_size, forecast_size]);
    }

    #[test]
    fn test_nbeats_parameter_count_positive() {
        let model = NBeats::new(
            /* backcast_size */ 10,
            /* forecast_size */ 5,
            /* num_stacks */ 2,
            /* num_blocks */ 3,
            /* hidden_size */ 16,
            /* num_fc_layers */ 2,
        );

        let params = model.parameters();
        assert!(!params.is_empty(), "model should have parameters");

        let total: usize = params.iter().map(|p| p.numel()).sum();
        assert!(total > 0, "total parameter count should be > 0");
    }

    #[test]
    fn test_nbeats_parameters_require_grad() {
        let model = NBeats::new(10, 5, 1, 1, 8, 1);
        for param in model.parameters() {
            assert!(
                param.requires_grad(),
                "all parameters should require grad"
            );
        }
    }

    #[test]
    fn test_nbeats_deterministic_with_same_input() {
        // Two forward passes with the same model + input should produce the
        // same output (no dropout / stochastic layers).
        let backcast_size = 6;
        let forecast_size = 2;

        let mut model = NBeats::new(backcast_size, forecast_size, 1, 1, 8, 1);

        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, backcast_size],
        )
        .expect("from_vec");

        let out1 = model.forward(&input).expect("forward 1");
        let out2 = model.forward(&input).expect("forward 2");

        let v1 = out1.to_vec();
        let v2 = out2.to_vec();
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "outputs should be identical: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_nbeats_accessors() {
        let model = NBeats::new(12, 4, 3, 2, 32, 3);
        assert_eq!(model.backcast_size(), 12);
        assert_eq!(model.forecast_size(), 4);
    }
}
