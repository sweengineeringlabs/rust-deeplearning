use llmforge::RuntimeConfig;
use llmforge::core::tensor::Tensor;

#[test]
fn runtime_config_default_apply_does_not_panic() {
    let config = RuntimeConfig::default();
    // num_threads=0 means auto-detect; should not panic
    let result = config.apply();
    assert!(result.is_ok());
}

#[test]
fn matmul_correct_after_parallelism_config() {
    // Apply config first
    let config = RuntimeConfig { num_threads: 0 };
    let _ = config.apply();

    // Create two known matrices and verify matmul result
    // A = [[1, 2], [3, 4]]  (2x2)
    // B = [[5, 6], [7, 8]]  (2x2)
    // A * B = [[19, 22], [43, 50]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

    let a_bytes = a_data.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>();
    let b_bytes = b_data.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>();

    let a = Tensor::new(a_bytes, vec![2, 2], llmforge::core::tensor::DType::F32);
    let b = Tensor::new(b_bytes, vec![2, 2], llmforge::core::tensor::DType::F32);

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);

    let c_data = c.as_slice_f32().unwrap();
    assert!((c_data[0] - 19.0).abs() < 1e-5);
    assert!((c_data[1] - 22.0).abs() < 1e-5);
    assert!((c_data[2] - 43.0).abs() < 1e-5);
    assert!((c_data[3] - 50.0).abs() < 1e-5);
}
