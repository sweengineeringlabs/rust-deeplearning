/// Feature engineering for OHLCV time series data.
///
/// FR-705: Pluggable feature trait and engine.
/// FR-706: Built-in technical indicators (Returns, SMA, Volatility).
/// FR-707: RSI indicator using Wilder's EMA.
use super::candle::OHLCVCandle;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A computed feature derived from OHLCV candle data.
pub trait Feature: Send + Sync {
    /// Compute the feature for each candle in `data`.
    ///
    /// The returned vector has the same length as `data`. Where the feature
    /// is undefined (e.g., first element for returns) it should be `0.0` or
    /// another sensible default.
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32>;

    /// Human-readable name for this feature.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Manages a collection of features and batch-computes them.
pub struct FeatureEngineer {
    features: Vec<Box<dyn Feature>>,
}

impl FeatureEngineer {
    /// Create an empty feature engineer.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Add a feature. Builder-style.
    pub fn add(mut self, feature: Box<dyn Feature>) -> Self {
        self.features.push(feature);
        self
    }

    /// Compute all features over the given candle data.
    ///
    /// Returns one `Vec<f32>` per feature, each of length `data.len()`.
    pub fn compute_all(&self, data: &[OHLCVCandle]) -> Vec<Vec<f32>> {
        self.features.iter().map(|f| f.compute(data)).collect()
    }

    /// Number of registered features.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether any features are registered.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in features
// ---------------------------------------------------------------------------

/// Log returns: `ln(close[i] / close[i-1])`.
///
/// The first element is `0.0` because there is no preceding candle.
pub struct Returns;

impl Feature for Returns {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        if data.is_empty() {
            return vec![];
        }
        let mut out = Vec::with_capacity(data.len());
        out.push(0.0);
        for i in 1..data.len() {
            let prev = data[i - 1].close;
            let curr = data[i].close;
            if prev > 0.0 {
                out.push((curr / prev).ln());
            } else {
                out.push(0.0);
            }
        }
        out
    }

    fn name(&self) -> &str {
        "returns"
    }
}

/// Simple Moving Average (SMA) of close prices over a rolling window.
///
/// Elements before the window is full are filled with the partial average.
pub struct MovingAverage {
    pub window: usize,
}

impl MovingAverage {
    pub fn new(window: usize) -> Self {
        Self { window }
    }
}

impl Feature for MovingAverage {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        if data.is_empty() || self.window == 0 {
            return vec![0.0; data.len()];
        }
        let mut out = Vec::with_capacity(data.len());
        let mut sum: f32 = 0.0;
        for (i, candle) in data.iter().enumerate() {
            sum += candle.close;
            if i >= self.window {
                sum -= data[i - self.window].close;
                out.push(sum / self.window as f32);
            } else {
                // Partial window: average of available elements
                out.push(sum / (i + 1) as f32);
            }
        }
        out
    }

    fn name(&self) -> &str {
        "moving_average"
    }
}

/// Rolling standard deviation of log returns (volatility).
///
/// Computed over a rolling window of log returns. Elements before the window
/// is full use a partial window. The very first element is `0.0`.
pub struct Volatility {
    pub window: usize,
}

impl Volatility {
    pub fn new(window: usize) -> Self {
        Self { window }
    }
}

impl Feature for Volatility {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        if data.is_empty() || self.window == 0 {
            return vec![0.0; data.len()];
        }

        // First compute log returns
        let returns = Returns.compute(data);

        let mut out = Vec::with_capacity(data.len());
        out.push(0.0); // First element has no return

        for i in 1..data.len() {
            let start = if i >= self.window { i - self.window + 1 } else { 1 };
            let window_returns = &returns[start..=i];
            let n = window_returns.len() as f32;
            if n <= 1.0 {
                out.push(0.0);
                continue;
            }
            let mean = window_returns.iter().sum::<f32>() / n;
            let variance =
                window_returns.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / (n - 1.0);
            out.push(variance.sqrt());
        }
        out
    }

    fn name(&self) -> &str {
        "volatility"
    }
}

/// Relative Strength Index using Wilder's Exponential Moving Average.
///
/// Implementation:
/// 1. Compute price changes: `delta[i] = close[i] - close[i-1]`.
/// 2. Seed the first average gain and loss from the simple mean of the
///    first `period` deltas.
/// 3. Subsequent values use Wilder's smoothing:
///    `avg_gain = avg_gain * (1 - alpha) + gain * alpha` where `alpha = 1/period`.
/// 4. `RSI = 100 - 100 / (1 + avg_gain / avg_loss)`.
///
/// Elements before `period` candles are available are set to `50.0` (neutral).
pub struct RSI {
    pub period: usize,
}

impl RSI {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Feature for RSI {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        let n = data.len();
        if n == 0 || self.period == 0 {
            return vec![50.0; n];
        }

        let mut out = vec![50.0; n];

        // Need at least period+1 data points (period deltas) to seed
        if n <= self.period {
            return out;
        }

        // Compute price deltas
        let deltas: Vec<f32> = (1..n).map(|i| data[i].close - data[i - 1].close).collect();

        // Seed: SMA of first `period` gains and losses
        let mut avg_gain: f32 = 0.0;
        let mut avg_loss: f32 = 0.0;
        for i in 0..self.period {
            let d = deltas[i];
            if d > 0.0 {
                avg_gain += d;
            } else {
                avg_loss += -d;
            }
        }
        avg_gain /= self.period as f32;
        avg_loss /= self.period as f32;

        // First RSI value at index = period (corresponds to delta index period-1)
        let rsi_val = if avg_loss == 0.0 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        };
        out[self.period] = rsi_val;

        // Wilder's smoothing for subsequent values
        let alpha = 1.0 / self.period as f32;
        for i in self.period..deltas.len() {
            let d = deltas[i];
            let gain = if d > 0.0 { d } else { 0.0 };
            let loss = if d < 0.0 { -d } else { 0.0 };

            avg_gain = avg_gain * (1.0 - alpha) + gain * alpha;
            avg_loss = avg_loss * (1.0 - alpha) + loss * alpha;

            let rsi_val = if avg_loss == 0.0 {
                100.0
            } else {
                100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            };
            // delta index i corresponds to data index i+1
            out[i + 1] = rsi_val;
        }

        out
    }

    fn name(&self) -> &str {
        "rsi"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles_with_prices(prices: &[f32]) -> Vec<OHLCVCandle> {
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| OHLCVCandle::new(i as i64, p, p + 1.0, p - 1.0, p, 1000.0))
            .collect()
    }

    #[test]
    fn test_returns_basic() {
        let candles = make_candles_with_prices(&[100.0, 110.0, 105.0]);
        let ret = Returns.compute(&candles);
        assert_eq!(ret.len(), 3);
        assert!((ret[0] - 0.0).abs() < 1e-6);
        assert!((ret[1] - (110.0_f32 / 100.0).ln()).abs() < 1e-5);
        assert!((ret[2] - (105.0_f32 / 110.0).ln()).abs() < 1e-5);
    }

    #[test]
    fn test_returns_empty() {
        let ret = Returns.compute(&[]);
        assert!(ret.is_empty());
    }

    #[test]
    fn test_returns_name() {
        assert_eq!(Returns.name(), "returns");
    }

    #[test]
    fn test_moving_average_basic() {
        let candles = make_candles_with_prices(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let ma = MovingAverage::new(3);
        let result = ma.compute(&candles);
        assert_eq!(result.len(), 5);

        // Partial windows
        assert!((result[0] - 10.0).abs() < 1e-6); // avg(10)
        assert!((result[1] - 15.0).abs() < 1e-6); // avg(10, 20)

        // Full window
        assert!((result[2] - 20.0).abs() < 1e-6); // avg(10, 20, 30)
        assert!((result[3] - 30.0).abs() < 1e-6); // avg(20, 30, 40)
        assert!((result[4] - 40.0).abs() < 1e-6); // avg(30, 40, 50)
    }

    #[test]
    fn test_moving_average_name() {
        assert_eq!(MovingAverage::new(5).name(), "moving_average");
    }

    #[test]
    fn test_volatility_basic() {
        // With constant prices, volatility should be 0
        let candles = make_candles_with_prices(&[100.0, 100.0, 100.0, 100.0, 100.0]);
        let vol = Volatility::new(3);
        let result = vol.compute(&candles);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-6);
        // All returns are 0, so all volatilities should be 0
        for v in &result {
            assert!(*v >= 0.0);
            assert!(*v < 1e-6);
        }
    }

    #[test]
    fn test_volatility_nonzero() {
        // With varying prices, volatility should be positive
        let candles = make_candles_with_prices(&[100.0, 110.0, 95.0, 108.0, 102.0]);
        let vol = Volatility::new(3);
        let result = vol.compute(&candles);
        // After the first element, volatility should be > 0
        assert!(result[2] > 0.0);
        assert!(result[3] > 0.0);
        assert!(result[4] > 0.0);
    }

    #[test]
    fn test_volatility_name() {
        assert_eq!(Volatility::new(5).name(), "volatility");
    }

    #[test]
    fn test_rsi_all_gains() {
        // Monotonically increasing prices should give RSI near 100
        let prices: Vec<f32> = (0..20).map(|i| 100.0 + i as f32).collect();
        let candles = make_candles_with_prices(&prices);
        let rsi = RSI::new(14);
        let result = rsi.compute(&candles);
        // After the seed period, RSI should be 100 (all gains, no losses)
        assert!((result[14] - 100.0).abs() < 1e-3);
        assert!((result[19] - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_rsi_all_losses() {
        // Monotonically decreasing prices should give RSI near 0
        let prices: Vec<f32> = (0..20).map(|i| 200.0 - i as f32).collect();
        let candles = make_candles_with_prices(&prices);
        let rsi = RSI::new(14);
        let result = rsi.compute(&candles);
        // After the seed period, RSI should be 0 (all losses, no gains)
        assert!((result[14] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_rsi_neutral() {
        // Before period is reached, RSI should be 50 (neutral default)
        let prices: Vec<f32> = (0..20).map(|i| 100.0 + (i as f32).sin() * 5.0).collect();
        let candles = make_candles_with_prices(&prices);
        let rsi = RSI::new(14);
        let result = rsi.compute(&candles);
        for i in 0..14 {
            assert!((result[i] - 50.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rsi_range() {
        // RSI should always be in [0, 100]
        let prices: Vec<f32> = (0..50)
            .map(|i| 100.0 + (i as f32 * 0.7).sin() * 20.0)
            .collect();
        let candles = make_candles_with_prices(&prices);
        let rsi = RSI::new(14);
        let result = rsi.compute(&candles);
        for val in &result {
            assert!(*val >= 0.0 && *val <= 100.0, "RSI out of range: {}", val);
        }
    }

    #[test]
    fn test_rsi_name() {
        assert_eq!(RSI::new(14).name(), "rsi");
    }

    #[test]
    fn test_feature_engineer() {
        let prices: Vec<f32> = (0..30).map(|i| 100.0 + i as f32 * 0.5).collect();
        let candles = make_candles_with_prices(&prices);

        let engine = FeatureEngineer::new()
            .add(Box::new(Returns))
            .add(Box::new(MovingAverage::new(5)))
            .add(Box::new(Volatility::new(5)))
            .add(Box::new(RSI::new(14)));

        assert_eq!(engine.len(), 4);
        assert!(!engine.is_empty());

        let results = engine.compute_all(&candles);
        assert_eq!(results.len(), 4);
        for result in &results {
            assert_eq!(result.len(), candles.len());
        }
    }

    #[test]
    fn test_feature_engineer_empty() {
        let engine = FeatureEngineer::new();
        assert!(engine.is_empty());
        assert_eq!(engine.len(), 0);
        let results = engine.compute_all(&[]);
        assert!(results.is_empty());
    }
}
