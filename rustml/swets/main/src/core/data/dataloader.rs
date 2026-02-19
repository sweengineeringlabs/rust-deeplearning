/// Batched data loader with optional shuffling for time series datasets.
///
/// FR-703: Batch iteration over `TimeSeriesDataset`.
/// FR-704: Optional shuffle of sample order.
use crate::api::tensor::Tensor;
use rand::seq::SliceRandom;
use rand::thread_rng;
use super::dataset::TimeSeriesDataset;

/// An iterator that yields batched `(input, target)` tensor pairs.
///
/// When `shuffle` is true, samples are yielded in a random order.
/// Each batch has:
/// - Input shape: `[batch_size, window_size, num_features]`
/// - Target shape: `[batch_size, target_dim]`
///
/// The last batch may be smaller than `batch_size` if the dataset length
/// is not evenly divisible.
pub struct DataLoader {
    dataset: TimeSeriesDataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_idx: usize,
}

impl DataLoader {
    /// Create a new `DataLoader`.
    ///
    /// If `shuffle` is true, sample indices are randomly permuted.
    pub fn new(dataset: TimeSeriesDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
        }
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_idx: 0,
        }
    }

    /// Reset the loader to the beginning, re-shuffling if configured.
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            let mut rng = thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Total number of batches per epoch (ceiling division).
    pub fn num_batches(&self) -> usize {
        let len = self.indices.len();
        if len == 0 {
            return 0;
        }
        (len + self.batch_size - 1) / self.batch_size
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end];
        self.current_idx = end;

        let actual_batch_size = batch_indices.len();
        let window_size = self.dataset.window_size();
        let num_features = self.dataset.num_features();
        let target_dim = self.dataset.target_dim();

        let mut input_data = Vec::with_capacity(actual_batch_size * window_size * num_features);
        let mut target_data = Vec::with_capacity(actual_batch_size * target_dim);

        for &idx in batch_indices {
            match self.dataset.get(idx) {
                Ok((input, target)) => {
                    input_data.extend_from_slice(&input.to_vec());
                    target_data.extend_from_slice(&target.to_vec());
                }
                Err(_) => {
                    // Skip invalid indices (should not happen with correct indices)
                    continue;
                }
            }
        }

        // Build batched tensors
        let input = Tensor::from_vec(
            input_data,
            vec![actual_batch_size, window_size, num_features],
        )
        .ok()?;

        let target = Tensor::from_vec(target_data, vec![actual_batch_size, target_dim]).ok()?;

        Some((input, target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::data::candle::OHLCVCandle;

    fn make_candles(n: usize) -> Vec<OHLCVCandle> {
        (0..n)
            .map(|i| {
                let v = i as f32;
                OHLCVCandle::new(i as i64, v, v + 1.0, v - 0.5, v + 0.5, (i * 100) as f32)
            })
            .collect()
    }

    #[test]
    fn test_dataloader_basic_iteration() {
        let ds = TimeSeriesDataset::new(make_candles(20), 3, 1);
        let expected_len = ds.len(); // 17
        let batch_size = 4;
        let loader = DataLoader::new(ds, batch_size, false);

        let batches: Vec<_> = loader.collect();
        // 17 samples / 4 = 4 full batches + 1 partial
        let expected_batches = (expected_len + batch_size - 1) / batch_size;
        assert_eq!(batches.len(), expected_batches);

        // First batch shape: [4, 3, 5]
        let (input, target) = &batches[0];
        assert_eq!(input.shape(), &[4, 3, 5]);
        assert_eq!(target.shape(), &[4, 1]);

        // Last batch may be smaller
        let (last_input, last_target) = &batches[batches.len() - 1];
        let last_batch_size = expected_len % batch_size;
        if last_batch_size > 0 {
            assert_eq!(last_input.shape()[0], last_batch_size);
            assert_eq!(last_target.shape()[0], last_batch_size);
        }
    }

    #[test]
    fn test_dataloader_shuffle_produces_all_samples() {
        let ds = TimeSeriesDataset::new(make_candles(15), 3, 1);
        let ds_len = ds.len();
        let loader = DataLoader::new(ds, 2, true);

        let mut total_samples = 0;
        for (input, _) in loader {
            total_samples += input.shape()[0];
        }
        assert_eq!(total_samples, ds_len);
    }

    #[test]
    fn test_dataloader_reset() {
        let ds = TimeSeriesDataset::new(make_candles(15), 3, 1);
        let mut loader = DataLoader::new(ds, 4, false);

        // Exhaust the loader
        let first_pass: Vec<_> = loader.by_ref().collect();
        assert!(!first_pass.is_empty());

        // Should be exhausted
        assert!(loader.next().is_none());

        // Reset and iterate again
        loader.reset();
        let second_pass: Vec<_> = loader.collect();
        assert_eq!(first_pass.len(), second_pass.len());
    }

    #[test]
    fn test_dataloader_num_batches() {
        let ds = TimeSeriesDataset::new(make_candles(20), 3, 1);
        let ds_len = ds.len();
        let loader = DataLoader::new(ds, 4, false);
        assert_eq!(loader.num_batches(), (ds_len + 3) / 4);
    }

    #[test]
    fn test_dataloader_multi_target() {
        let ds = TimeSeriesDataset::new(make_candles(15), 3, 1).with_targets(
            crate::core::data::dataset::TargetColumn::Multi(vec![
                "open".to_string(),
                "close".to_string(),
            ]),
        );
        let loader = DataLoader::new(ds, 4, false);

        let (_, target) = loader.into_iter().next().unwrap();
        assert_eq!(target.shape()[1], 2); // 2 target columns
    }

    #[test]
    fn test_dataloader_empty_dataset() {
        let ds = TimeSeriesDataset::new(make_candles(2), 5, 1);
        assert_eq!(ds.len(), 0);
        let loader = DataLoader::new(ds, 4, false);
        assert_eq!(loader.num_batches(), 0);
        let batches: Vec<_> = loader.collect();
        assert!(batches.is_empty());
    }
}
