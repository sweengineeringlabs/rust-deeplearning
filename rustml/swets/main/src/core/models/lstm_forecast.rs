use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tensor::Tensor;
use crate::core::nn::linear::Linear;
use crate::core::nn::lstm::LSTM;

/// LSTM-based forecasting model (FR-804).
///
/// Architecture:
///   1. Multi-layer LSTM processes input sequence: [batch, seq_len, input_size] -> [batch, seq_len, hidden_size]
///   2. Take the last timestep's hidden state: [batch, hidden_size]
///   3. Fully-connected layer projects to output: [batch, output_size]
///
/// Typical use: time-series forecasting where the model reads a window of
/// historical features and produces a prediction (e.g., next-step value).
pub struct LSTMForecast {
    lstm: LSTM,
    fc: Linear,
}

impl LSTMForecast {
    /// Creates a new LSTM forecasting model.
    ///
    /// - `input_size`: number of input features per timestep
    /// - `hidden_size`: LSTM hidden dimension
    /// - `num_layers`: number of stacked LSTM layers
    /// - `output_size`: dimensionality of the forecast output
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, output_size: usize) -> Self {
        let lstm = LSTM::new(input_size, hidden_size, num_layers);
        let fc = Linear::new(hidden_size, output_size);
        Self { lstm, fc }
    }

    /// Resets the LSTM's internal hidden and cell states.
    /// Should be called between independent sequences during inference.
    pub fn reset_state(&mut self) {
        self.lstm.reset_state();
    }

    /// Returns a reference to the inner LSTM layer.
    pub fn lstm(&self) -> &LSTM {
        &self.lstm
    }

    /// Returns a reference to the inner fully-connected layer.
    pub fn fc(&self) -> &Linear {
        &self.fc
    }
}

impl Layer for LSTMForecast {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        // input: [batch, seq_len, input_size]
        let lstm_out = self.lstm.forward(input)?;
        // lstm_out: [batch, seq_len, hidden_size]

        let shape = lstm_out.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];

        // Extract the last timestep: [batch, hidden_size]
        let all_data = lstm_out.to_vec();
        let mut last_step_data = vec![0.0f32; batch * hidden_size];
        for b in 0..batch {
            let src_start = b * seq_len * hidden_size + (seq_len - 1) * hidden_size;
            let dst_start = b * hidden_size;
            last_step_data[dst_start..dst_start + hidden_size]
                .copy_from_slice(&all_data[src_start..src_start + hidden_size]);
        }
        let last_hidden = Tensor::from_vec(last_step_data, vec![batch, hidden_size])?;

        // Project through fully connected layer: [batch, hidden_size] -> [batch, output_size]
        let output = self.fc.forward(&last_hidden)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.lstm.parameters();
        params.extend(self.fc.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.lstm.parameters_mut();
        params.extend(self.fc.parameters_mut());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_forecast_output_shape() {
        let mut model = LSTMForecast::new(5, 16, 2, 1);
        let input = Tensor::randn([8, 10, 5]); // batch=8, seq_len=10, features=5
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8, 1]); // batch=8, output_size=1
    }

    #[test]
    fn test_lstm_forecast_multi_output() {
        let mut model = LSTMForecast::new(3, 8, 1, 4);
        let input = Tensor::randn([2, 5, 3]); // batch=2, seq_len=5, features=3
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 4]); // batch=2, output_size=4
    }

    #[test]
    fn test_lstm_forecast_parameters() {
        let model = LSTMForecast::new(5, 16, 2, 1);
        let params = model.parameters();
        // LSTM: 2 layers * 4 = 8 params, FC: weight + bias = 2 params
        assert_eq!(params.len(), 10);
    }

    #[test]
    fn test_lstm_forecast_reset() {
        let mut model = LSTMForecast::new(3, 8, 1, 1);
        let input = Tensor::randn([1, 4, 3]);

        let out1 = model.forward(&input).unwrap();
        model.reset_state();
        let out2 = model.forward(&input).unwrap();

        let d1 = out1.to_vec();
        let d2 = out2.to_vec();
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "After reset, outputs should match: {} vs {}",
                a,
                b
            );
        }
    }
}
