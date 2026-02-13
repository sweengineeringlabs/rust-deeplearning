use llmforge::core::arena::TensorPool;
use llmforge::core::tensor::{Tensor, DType, f32_vec_to_bytes};

#[test]
fn pool_get_returns_correct_size() {
    let mut pool = TensorPool::new(10);
    let buf = pool.get(1024);
    assert_eq!(buf.len(), 1024);
    assert!(pool.is_empty()); // Nothing returned yet
}

#[test]
fn pool_put_and_reuse() {
    let mut pool = TensorPool::new(10);

    // Get and return a buffer
    let buf = pool.get(1024);
    assert_eq!(buf.len(), 1024);
    let cap = buf.capacity();
    pool.put(buf);
    assert_eq!(pool.len(), 1);

    // Get again - should reuse
    let buf2 = pool.get(512);
    assert_eq!(buf2.len(), 512);
    assert!(buf2.capacity() >= cap); // Reused buffer has at least original capacity
    assert_eq!(pool.len(), 0); // Buffer removed from pool
}

#[test]
fn pool_capacity_limits() {
    let mut pool = TensorPool::new(2);

    pool.put(vec![0u8; 100]);
    pool.put(vec![0u8; 200]);
    assert_eq!(pool.len(), 2);

    // Third put should be dropped (capacity = 2)
    pool.put(vec![0u8; 300]);
    assert_eq!(pool.len(), 2);
}

#[test]
fn tensor_into_bytes_owned() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes = f32_vec_to_bytes(data);
    let expected_len = bytes.len();
    let tensor = Tensor::new(bytes, vec![4], DType::F32);

    let extracted = tensor.into_bytes();
    assert!(extracted.is_some());
    assert_eq!(extracted.unwrap().len(), expected_len);
}

#[test]
fn tensor_into_bytes_shared_returns_none() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes = f32_vec_to_bytes(data);
    let tensor = Tensor::new(bytes, vec![4], DType::F32);
    let _clone = tensor.clone(); // Creates shared Arc

    let extracted = tensor.into_bytes();
    assert!(extracted.is_none());
}
