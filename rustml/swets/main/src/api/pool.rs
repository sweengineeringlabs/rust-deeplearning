use std::cell::RefCell;
use std::collections::HashMap;

/// Thread-local tensor buffer pool for recycling `Vec<f32>` allocations
/// during the backward pass (FR-104).
///
/// Buffers are bucketed by capacity (rounded up to the next power of 2)
/// so that a returned buffer can satisfy any request up to that bucket size.
struct TensorPool {
    /// Bins keyed by power-of-2 capacity, each holding a stack of reusable buffers.
    bins: HashMap<usize, Vec<Vec<f32>>>,
}

impl TensorPool {
    fn new() -> Self {
        Self {
            bins: HashMap::new(),
        }
    }

    /// Acquire a zeroed buffer with at least `size` elements.
    ///
    /// If a recycled buffer exists in the appropriate bucket it is reused
    /// (cleared to zero); otherwise a fresh allocation is returned.
    fn acquire(&mut self, size: usize) -> Vec<f32> {
        let bucket = size.next_power_of_two().max(1);
        if let Some(stack) = self.bins.get_mut(&bucket) {
            if let Some(mut buf) = stack.pop() {
                buf.clear();
                buf.resize(size, 0.0);
                return buf;
            }
        }
        vec![0.0; size]
    }

    /// Return a buffer to the pool for later reuse.
    fn release(&mut self, buf: Vec<f32>) {
        let bucket = buf.capacity().next_power_of_two().max(1);
        self.bins.entry(bucket).or_default().push(buf);
    }

    /// Drain every cached buffer from all bins.
    fn clear(&mut self) {
        self.bins.clear();
    }
}

// --- Thread-local API ---

thread_local! {
    static POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new());
}

/// Acquire a zeroed `Vec<f32>` of at least `size` elements, reusing a
/// previously released buffer when one is available.
pub fn acquire(size: usize) -> Vec<f32> {
    POOL.with(|pool| pool.borrow_mut().acquire(size))
}

/// Return a buffer to the thread-local pool so it can be recycled.
pub fn release(buf: Vec<f32>) {
    POOL.with(|pool| pool.borrow_mut().release(buf));
}

/// Drain all cached buffers from the thread-local pool.
pub fn clear_pool() {
    POOL.with(|pool| pool.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_returns_zeroed_buffer() {
        let buf = acquire(10);
        assert_eq!(buf.len(), 10);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn release_and_reacquire_reuses_allocation() {
        let buf = acquire(8);
        let ptr = buf.as_ptr();
        release(buf);

        // Same bucket (8 -> next_power_of_two = 8), should get the same allocation back
        let buf2 = acquire(8);
        assert_eq!(buf2.as_ptr(), ptr);
        assert_eq!(buf2.len(), 8);
        assert!(buf2.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn clear_pool_drains_all_bins() {
        release(acquire(16));
        release(acquire(64));
        clear_pool();

        // After clear, acquire should return fresh allocations (no reuse test,
        // just make sure it doesn't panic).
        let buf = acquire(16);
        assert_eq!(buf.len(), 16);
    }

    #[test]
    fn acquire_zero_size() {
        let buf = acquire(0);
        assert!(buf.is_empty());
    }
}
