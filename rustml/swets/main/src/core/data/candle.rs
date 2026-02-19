/// OHLCV candle data point representing a single time period.
///
/// FR-700: Core data structure for financial time series data.
#[derive(Debug, Clone)]
pub struct OHLCVCandle {
    /// Unix timestamp (seconds since epoch).
    pub timestamp: i64,
    /// Opening price.
    pub open: f32,
    /// Highest price during the period.
    pub high: f32,
    /// Lowest price during the period.
    pub low: f32,
    /// Closing price.
    pub close: f32,
    /// Volume traded during the period.
    pub volume: f32,
}

impl OHLCVCandle {
    /// Create a new OHLCV candle.
    pub fn new(timestamp: i64, open: f32, high: f32, low: f32, close: f32, volume: f32) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Extract a feature value by column name.
    ///
    /// Returns `None` if the column name is not recognized.
    pub fn get_feature(&self, name: &str) -> Option<f32> {
        match name {
            "open" => Some(self.open),
            "high" => Some(self.high),
            "low" => Some(self.low),
            "close" => Some(self.close),
            "volume" => Some(self.volume),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_construction() {
        let candle = OHLCVCandle::new(1_700_000_000, 100.0, 105.0, 99.0, 103.0, 50_000.0);
        assert_eq!(candle.timestamp, 1_700_000_000);
        assert_eq!(candle.open, 100.0);
        assert_eq!(candle.high, 105.0);
        assert_eq!(candle.low, 99.0);
        assert_eq!(candle.close, 103.0);
        assert_eq!(candle.volume, 50_000.0);
    }

    #[test]
    fn test_candle_clone() {
        let candle = OHLCVCandle::new(1_700_000_000, 100.0, 105.0, 99.0, 103.0, 50_000.0);
        let cloned = candle.clone();
        assert_eq!(candle.timestamp, cloned.timestamp);
        assert_eq!(candle.close, cloned.close);
    }

    #[test]
    fn test_get_feature() {
        let candle = OHLCVCandle::new(0, 10.0, 20.0, 5.0, 15.0, 1000.0);
        assert_eq!(candle.get_feature("open"), Some(10.0));
        assert_eq!(candle.get_feature("high"), Some(20.0));
        assert_eq!(candle.get_feature("low"), Some(5.0));
        assert_eq!(candle.get_feature("close"), Some(15.0));
        assert_eq!(candle.get_feature("volume"), Some(1000.0));
        assert_eq!(candle.get_feature("unknown"), None);
    }
}
