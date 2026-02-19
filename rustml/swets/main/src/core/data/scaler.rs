/// Feature scaling / normalization for time series data.
///
/// FR-708: Fit scalers on training data.
/// FR-709: Transform and inverse-transform for inference.

/// The type of normalization to apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalerType {
    /// Min-Max normalization to [0, 1]:  `(x - min) / (max - min)`
    MinMax,
    /// Standardization (z-score):  `(x - mean) / std`
    Standard,
    /// Robust scaling using median and IQR:  `(x - median) / (Q75 - Q25)`
    Robust,
}

/// A per-feature scaler that can fit, transform, and inverse-transform data.
///
/// Internally stores a pair of parameters `(p1, p2)` per feature column:
/// - `MinMax`: `(min, range)` where `range = max - min`
/// - `Standard`: `(mean, std)`
/// - `Robust`: `(median, iqr)` where `iqr = Q75 - Q25`
#[derive(Debug, Clone)]
pub struct Scaler {
    scaler_type: ScalerType,
    params: Vec<(f32, f32)>,
}

impl Scaler {
    /// Fit scaler parameters from the given data.
    ///
    /// `data` is a slice of feature columns, where each inner `Vec<f32>` is one
    /// column (all values for a single feature across all samples).
    pub fn fit(data: &[Vec<f32>], scaler_type: ScalerType) -> Self {
        let params: Vec<(f32, f32)> = data
            .iter()
            .map(|col| match scaler_type {
                ScalerType::MinMax => {
                    let min = col.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let range = max - min;
                    (min, if range == 0.0 { 1.0 } else { range })
                }
                ScalerType::Standard => {
                    let n = col.len() as f32;
                    if n == 0.0 {
                        return (0.0, 1.0);
                    }
                    let mean = col.iter().sum::<f32>() / n;
                    let variance = col.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                    let std = variance.sqrt();
                    (mean, if std == 0.0 { 1.0 } else { std })
                }
                ScalerType::Robust => {
                    if col.is_empty() {
                        return (0.0, 1.0);
                    }
                    let median = compute_median(col);
                    let q25 = compute_percentile(col, 0.25);
                    let q75 = compute_percentile(col, 0.75);
                    let iqr = q75 - q25;
                    (median, if iqr == 0.0 { 1.0 } else { iqr })
                }
            })
            .collect();

        Self {
            scaler_type,
            params,
        }
    }

    /// Transform data using the fitted parameters.
    ///
    /// `data` layout matches `fit`: one `Vec<f32>` per feature column.
    /// Returns transformed columns in the same layout.
    pub fn transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        data.iter()
            .zip(self.params.iter())
            .map(|(col, &(p1, p2))| col.iter().map(|&x| (x - p1) / p2).collect())
            .collect()
    }

    /// Inverse-transform data back to the original scale.
    ///
    /// `data` layout matches `fit`: one `Vec<f32>` per feature column.
    pub fn inverse_transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        data.iter()
            .zip(self.params.iter())
            .map(|(col, &(p1, p2))| col.iter().map(|&x| x * p2 + p1).collect())
            .collect()
    }

    /// The scaler type.
    pub fn scaler_type(&self) -> ScalerType {
        self.scaler_type
    }

    /// Per-feature parameters.
    pub fn params(&self) -> &[(f32, f32)] {
        &self.params
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the median of a slice (creates a sorted copy).
fn compute_median(values: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute a percentile (0.0..1.0) using linear interpolation.
fn compute_percentile(values: &[f32], percentile: f32) -> f32 {
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = percentile * (n - 1) as f32;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f32;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_fit_transform() {
        let data = vec![vec![0.0, 10.0, 20.0, 30.0, 40.0]];
        let scaler = Scaler::fit(&data, ScalerType::MinMax);

        let transformed = scaler.transform(&data);
        assert_eq!(transformed.len(), 1);
        assert!((transformed[0][0] - 0.0).abs() < 1e-6);
        assert!((transformed[0][4] - 1.0).abs() < 1e-6);
        assert!((transformed[0][2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_inverse_transform() {
        let data = vec![vec![0.0, 10.0, 20.0, 30.0, 40.0]];
        let scaler = Scaler::fit(&data, ScalerType::MinMax);
        let transformed = scaler.transform(&data);
        let recovered = scaler.inverse_transform(&transformed);

        for (orig, rec) in data[0].iter().zip(recovered[0].iter()) {
            assert!((orig - rec).abs() < 1e-5);
        }
    }

    #[test]
    fn test_standard_fit_transform() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let scaler = Scaler::fit(&data, ScalerType::Standard);
        let transformed = scaler.transform(&data);

        // Mean of transformed should be ~0
        let mean: f32 = transformed[0].iter().sum::<f32>() / transformed[0].len() as f32;
        assert!(mean.abs() < 1e-5);

        // Std of transformed should be ~1 (population std)
        let var: f32 = transformed[0].iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / transformed[0].len() as f32;
        assert!((var.sqrt() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_standard_inverse_transform() {
        let data = vec![vec![10.0, 20.0, 30.0, 40.0, 50.0]];
        let scaler = Scaler::fit(&data, ScalerType::Standard);
        let transformed = scaler.transform(&data);
        let recovered = scaler.inverse_transform(&transformed);

        for (orig, rec) in data[0].iter().zip(recovered[0].iter()) {
            assert!((orig - rec).abs() < 1e-4);
        }
    }

    #[test]
    fn test_robust_fit_transform() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let scaler = Scaler::fit(&data, ScalerType::Robust);

        let params = scaler.params();
        // Median of 1..10 = 5.5
        assert!((params[0].0 - 5.5).abs() < 1e-5);
        // IQR: Q75 - Q25
        let q25 = compute_percentile(&data[0], 0.25);
        let q75 = compute_percentile(&data[0], 0.75);
        assert!((params[0].1 - (q75 - q25)).abs() < 1e-5);
    }

    #[test]
    fn test_robust_inverse_transform() {
        let data = vec![vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]];
        let scaler = Scaler::fit(&data, ScalerType::Robust);
        let transformed = scaler.transform(&data);
        let recovered = scaler.inverse_transform(&transformed);

        for (orig, rec) in data[0].iter().zip(recovered[0].iter()) {
            assert!((orig - rec).abs() < 1e-4);
        }
    }

    #[test]
    fn test_multi_feature_columns() {
        let data = vec![
            vec![0.0, 10.0, 20.0, 30.0],
            vec![100.0, 200.0, 300.0, 400.0],
        ];
        let scaler = Scaler::fit(&data, ScalerType::MinMax);
        let transformed = scaler.transform(&data);
        let recovered = scaler.inverse_transform(&transformed);

        // First feature
        assert!((transformed[0][0] - 0.0).abs() < 1e-6);
        assert!((transformed[0][3] - 1.0).abs() < 1e-6);

        // Second feature
        assert!((transformed[1][0] - 0.0).abs() < 1e-6);
        assert!((transformed[1][3] - 1.0).abs() < 1e-6);

        // Roundtrip
        for (col_orig, col_rec) in data.iter().zip(recovered.iter()) {
            for (orig, rec) in col_orig.iter().zip(col_rec.iter()) {
                assert!((orig - rec).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_constant_column_handling() {
        // All values the same - should not divide by zero
        let data = vec![vec![5.0, 5.0, 5.0, 5.0]];

        let scaler = Scaler::fit(&data, ScalerType::MinMax);
        let transformed = scaler.transform(&data);
        for &v in &transformed[0] {
            assert!(v.is_finite());
        }

        let scaler = Scaler::fit(&data, ScalerType::Standard);
        let transformed = scaler.transform(&data);
        for &v in &transformed[0] {
            assert!(v.is_finite());
        }

        let scaler = Scaler::fit(&data, ScalerType::Robust);
        let transformed = scaler.transform(&data);
        for &v in &transformed[0] {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<f32>> = vec![];
        let scaler = Scaler::fit(&data, ScalerType::MinMax);
        assert!(scaler.params().is_empty());
        let transformed = scaler.transform(&data);
        assert!(transformed.is_empty());
    }

    #[test]
    fn test_scaler_type_accessor() {
        let scaler = Scaler::fit(&[vec![1.0, 2.0]], ScalerType::Standard);
        assert_eq!(scaler.scaler_type(), ScalerType::Standard);
    }

    #[test]
    fn test_median_even() {
        assert!((compute_median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_median_odd() {
        assert!((compute_median(&[1.0, 3.0, 2.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_percentile_boundaries() {
        let vals = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert!((compute_percentile(&vals, 0.0) - 10.0).abs() < 1e-6);
        assert!((compute_percentile(&vals, 1.0) - 50.0).abs() < 1e-6);
        assert!((compute_percentile(&vals, 0.5) - 30.0).abs() < 1e-6);
    }
}
