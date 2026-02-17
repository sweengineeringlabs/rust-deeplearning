//! KV Cache for efficient autoregressive inference.
//!
//! Pre-allocates key/value buffers for each transformer layer and provides
//! efficient update/view operations for incremental decoding.

use crate::api::error::{NnError, NnResult};
use rustml_core::Tensor;

/// KV Cache storing past keys and values for each layer.
pub struct KVCache {
    past_keys: Vec<Tensor>,
    past_values: Vec<Tensor>,
    max_seq_len: usize,
    pub current_len: usize,
    head_dim: usize,
    num_kv_heads: usize,
}

impl KVCache {
    /// Create a new KV cache.
    ///
    /// Pre-allocates [1, num_kv_heads, max_seq_len, head_dim] buffers per layer.
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        head_dim: usize,
        num_kv_heads: usize,
    ) -> Self {
        let key_shape = vec![1, num_kv_heads, max_seq_len, head_dim];
        let val_shape = vec![1, num_kv_heads, max_seq_len, head_dim];

        let past_keys = (0..num_layers)
            .map(|_| Tensor::zeros(key_shape.clone()))
            .collect();
        let past_values = (0..num_layers)
            .map(|_| Tensor::zeros(val_shape.clone()))
            .collect();

        Self {
            past_keys,
            past_values,
            max_seq_len,
            current_len: 0,
            head_dim,
            num_kv_heads,
        }
    }

    /// Create a KV cache with a parameterized batch size (for batched inference).
    pub fn new_batched(
        num_layers: usize,
        max_seq_len: usize,
        head_dim: usize,
        num_kv_heads: usize,
        batch_size: usize,
    ) -> Self {
        let key_shape = vec![batch_size, num_kv_heads, max_seq_len, head_dim];
        let val_shape = vec![batch_size, num_kv_heads, max_seq_len, head_dim];

        let past_keys = (0..num_layers)
            .map(|_| Tensor::zeros(key_shape.clone()))
            .collect();
        let past_values = (0..num_layers)
            .map(|_| Tensor::zeros(val_shape.clone()))
            .collect();

        Self {
            past_keys,
            past_values,
            max_seq_len,
            current_len: 0,
            head_dim,
            num_kv_heads,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get a view of cached K/V for `0..len` on the sequence dimension.
    pub fn get_view(&self, layer_idx: usize, len: usize) -> NnResult<(Tensor, Tensor)> {
        let k = self.past_keys[layer_idx].slice_sequence(0, len)?;
        let v = self.past_values[layer_idx].slice_sequence(0, len)?;
        Ok((k, v))
    }

    /// Write new K/V entries into the cache at `current_len`.
    pub fn update(&mut self, layer_idx: usize, key: Tensor, value: Tensor) -> NnResult<()> {
        let seq_len = key.shape()[2];
        if self.current_len + seq_len > self.max_seq_len {
            return Err(NnError::InvalidConfig(format!(
                "Sequence length exceeded: max={}, actual={}",
                self.max_seq_len,
                self.current_len + seq_len,
            )));
        }
        self.past_keys[layer_idx].slice_assign_sequence(self.current_len, &key)?;
        self.past_values[layer_idx].slice_assign_sequence(self.current_len, &value)?;
        Ok(())
    }

    /// Advance the current position by `step` tokens.
    pub fn advance(&mut self, step: usize) {
        self.current_len += step;
    }

    /// Create a deep copy with independently owned tensor data (not Arc-shared).
    /// Required for beam search where each beam needs a mutable cache.
    pub fn deep_clone(&self) -> NnResult<Self> {
        let past_keys = self
            .past_keys
            .iter()
            .map(|t| {
                let bytes = t.as_raw_bytes()?.to_vec();
                Ok(Tensor::new(bytes, t.shape().to_vec(), t.dtype()))
            })
            .collect::<NnResult<Vec<_>>>()?;

        let past_values = self
            .past_values
            .iter()
            .map(|t| {
                let bytes = t.as_raw_bytes()?.to_vec();
                Ok(Tensor::new(bytes, t.shape().to_vec(), t.dtype()))
            })
            .collect::<NnResult<Vec<_>>>()?;

        Ok(Self {
            past_keys,
            past_values,
            max_seq_len: self.max_seq_len,
            current_len: self.current_len,
            head_dim: self.head_dim,
            num_kv_heads: self.num_kv_heads,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_basic() {
        let mut cache = KVCache::new(2, 32, 64, 4);
        assert_eq!(cache.current_len, 0);
        assert_eq!(cache.head_dim(), 64);

        // Insert [1, 4, 5, 64] K/V into layer 0
        let k = Tensor::randn(vec![1, 4, 5, 64]);
        let v = Tensor::randn(vec![1, 4, 5, 64]);
        cache.update(0, k, v).unwrap();
        cache.advance(5);
        assert_eq!(cache.current_len, 5);

        let (k_view, v_view) = cache.get_view(0, 5).unwrap();
        assert_eq!(k_view.shape()[2], 5);
        assert_eq!(v_view.shape()[2], 5);
    }

    #[test]
    fn test_kv_cache_overflow() {
        let mut cache = KVCache::new(1, 4, 8, 2);
        let k = Tensor::randn(vec![1, 2, 5, 8]);
        let v = Tensor::randn(vec![1, 2, 5, 8]);
        // seq_len=5 > max_seq_len=4 should fail
        assert!(cache.update(0, k, v).is_err());
    }

    #[test]
    fn test_kv_cache_deep_clone() {
        let mut cache = KVCache::new(1, 16, 8, 2);
        let k = Tensor::randn(vec![1, 2, 3, 8]);
        let v = Tensor::randn(vec![1, 2, 3, 8]);
        cache.update(0, k, v).unwrap();
        cache.advance(3);

        let clone = cache.deep_clone().unwrap();
        assert_eq!(clone.current_len, 3);
    }
}
