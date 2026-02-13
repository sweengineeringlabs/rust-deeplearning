mod common;

use common::{make_f32_tensor, assert_f32_near};

#[test]
fn matmul_2x2_known_product() {
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = make_f32_tensor(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = make_f32_tensor(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    let data = c.as_slice_f32().unwrap();
    assert_f32_near(data, &[19.0, 22.0, 43.0, 50.0], 1e-5, "2x2 matmul");
}

#[test]
fn matmul_2x3_times_3x2() {
    // [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]]
    // = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[58,64],[139,154]]
    let a = make_f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = make_f32_tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    let data = c.as_slice_f32().unwrap();
    assert_f32_near(data, &[58.0, 64.0, 139.0, 154.0], 1e-4, "2x3 * 3x2");
}

#[test]
fn matmul_shape_mismatch_returns_error() {
    let a = make_f32_tensor(&[0.0; 6], vec![2, 3]);
    let b = make_f32_tensor(&[0.0; 8], vec![4, 2]);
    let result = a.matmul(&b);
    assert!(result.is_err(), "Expected shape mismatch error for [2,3]*[4,2]");
}

#[test]
fn matmul_3d_broadcasts_correctly() {
    // [2,3,4] * [4,5] -> [2,3,5] with broadcasting
    // Use identity-like values for verification
    let a_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let a = make_f32_tensor(&a_data, vec![2, 3, 4]);

    // Identity-ish: 4x5 matrix where each row has one 1.0 (first 4 cols of I_5)
    let mut b_data = vec![0.0f32; 20];
    for i in 0..4 {
        b_data[i * 5 + i] = 1.0;
    }
    let b = make_f32_tensor(&b_data, vec![4, 5]);

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 3, 5]);

    // Each output row should be [a[row][0], a[row][1], a[row][2], a[row][3], 0.0]
    let data = c.as_slice_f32().unwrap();
    // First row of first batch: a=[0,1,2,3] -> c=[0,1,2,3,0]
    assert_f32_near(&data[0..5], &[0.0, 1.0, 2.0, 3.0, 0.0], 1e-5, "3d broadcast row 0");
}

#[test]
fn batched_matmul_3d_known_product() {
    // Batch=2, each [2,3] * [3,2]
    // Batch 0: [[1,2,3],[4,5,6]] * [[1,0],[0,1],[1,1]] = [[4,5],[10,11]]
    // Batch 1: [[2,0,0],[0,2,0]] * [[1,0],[0,1],[0,0]] = [[2,0],[0,2]]
    let a = make_f32_tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        vec![2, 2, 3],
    );
    let b = make_f32_tensor(
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![2, 3, 2],
    );
    let c = a.batched_matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2, 2]);
    let data = c.as_slice_f32().unwrap();
    assert_f32_near(
        data,
        &[4.0, 5.0, 10.0, 11.0, 2.0, 0.0, 0.0, 2.0],
        1e-5,
        "batched 3d matmul",
    );
}

#[test]
fn softmax_uniform_input() {
    let t = make_f32_tensor(&[1.0, 1.0, 1.0, 1.0], vec![1, 4]);
    let s = t.softmax(-1).unwrap();
    let data = s.as_slice_f32().unwrap();
    assert_f32_near(data, &[0.25, 0.25, 0.25, 0.25], 1e-6, "uniform softmax");
}

#[test]
fn softmax_large_values_stable() {
    let t = make_f32_tensor(&[1000.0, 1000.0, 1000.0], vec![1, 3]);
    let s = t.softmax(-1).unwrap();
    let data = s.as_slice_f32().unwrap();
    for &v in data {
        assert!(!v.is_nan(), "softmax produced NaN for large values");
    }
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum should be 1.0, got {}", sum);
}

#[test]
fn softmax_negative_values() {
    let t = make_f32_tensor(&[-100.0, 0.0, 100.0], vec![1, 3]);
    let s = t.softmax(-1).unwrap();
    let data = s.as_slice_f32().unwrap();
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum should be 1.0, got {}", sum);
    // Largest input should get largest probability
    assert!(data[2] > data[1], "softmax(100) should be > softmax(0)");
    assert!(data[1] > data[0], "softmax(0) should be > softmax(-100)");
}

#[test]
fn softmax_single_element() {
    let t = make_f32_tensor(&[42.0], vec![1, 1]);
    let s = t.softmax(-1).unwrap();
    let data = s.as_slice_f32().unwrap();
    assert_f32_near(data, &[1.0], 1e-6, "single element softmax");
}

#[test]
fn layer_norm_zero_mean_unit_variance() {
    // Input with known statistics, identity weight (gamma=1), zero bias (beta=0)
    let input = make_f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let weight = make_f32_tensor(&[1.0, 1.0, 1.0, 1.0], vec![4]);
    let bias = make_f32_tensor(&[0.0, 0.0, 0.0, 0.0], vec![4]);

    let out = input.layer_norm(&weight, &bias, 1e-5).unwrap();
    let data = out.as_slice_f32().unwrap();

    // Each row should have mean ~0 and variance ~1
    for row in 0..2 {
        let start = row * 4;
        let row_data = &data[start..start + 4];
        let mean: f32 = row_data.iter().sum::<f32>() / 4.0;
        let var: f32 = row_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-4,
            "row {} mean should be ~0, got {}",
            row,
            mean
        );
        assert!(
            (var - 1.0).abs() < 0.1,
            "row {} variance should be ~1, got {}",
            row,
            var
        );
    }
}

#[test]
fn layer_norm_with_weight_and_bias() {
    // Single row: [2, 4, 6, 8] -> mean=5, var=5
    // Normalized: (x-5)/sqrt(5+eps) -> [-1.3416, -0.4472, 0.4472, 1.3416] (approx)
    // With gamma=2, beta=1: result = 2*norm + 1
    let input = make_f32_tensor(&[2.0, 4.0, 6.0, 8.0], vec![1, 4]);
    let weight = make_f32_tensor(&[2.0, 2.0, 2.0, 2.0], vec![4]);
    let bias = make_f32_tensor(&[1.0, 1.0, 1.0, 1.0], vec![4]);

    let out = input.layer_norm(&weight, &bias, 1e-5).unwrap();
    let data = out.as_slice_f32().unwrap();

    // mean=5, var=(9+1+1+9)/4=5, std=sqrt(5+1e-5)≈2.23607
    // norm = (x-5)/2.23607 => [-1.3416, -0.4472, 0.4472, 1.3416]
    // result = 2*norm + 1 => [-1.6833, 0.1056, 1.8944, 3.6833]
    let std_val = (5.0f32 + 1e-5).sqrt();
    let expected: Vec<f32> = [2.0, 4.0, 6.0, 8.0]
        .iter()
        .map(|&x| 2.0 * (x - 5.0) / std_val + 1.0)
        .collect();
    assert_f32_near(data, &expected, 1e-4, "layer_norm with weight and bias");
}
