mod common;

use common::{make_f32_tensor, assert_f32_near};
use llmforge::nn::{Linear, Embedding, LayerNorm, Layer};

#[test]
fn linear_forward_known_weights() {
    // W = [[1,0],[0,1],[1,1]] (3x2), input = [3,7] (1x2)
    // Forward: output = input @ W^T
    // W^T = [[1,0,1],[0,1,1]]
    // output = [3,7] @ [[1,0,1],[0,1,1]] = [3*1+7*0, 3*0+7*1, 3*1+7*1] = [3, 7, 10]
    let weight = make_f32_tensor(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
    let linear = Linear::from_weights(weight, None);

    let input = make_f32_tensor(&[3.0, 7.0], vec![1, 2]);
    let output = linear.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    let data = output.as_slice_f32().unwrap();
    assert_f32_near(data, &[3.0, 7.0, 10.0], 1e-4, "linear forward known");
}

#[test]
fn linear_forward_3d_input() {
    // W = [4, 3], input = [2, 3, 3] -> output = [2, 3, 4]
    let weight = make_f32_tensor(&[0.0; 12], vec![4, 3]);
    let linear = Linear::from_weights(weight, None);

    let input = make_f32_tensor(&[0.0; 18], vec![2, 3, 3]);
    let output = linear.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 3, 4]);
}

#[test]
fn linear_with_bias_adds_correctly() {
    // W = [[1,0],[0,1]] (identity 2x2), bias = [10, 20]
    // input = [3, 7] -> output = [3, 7] + [10, 20] = [13, 27]
    let weight = make_f32_tensor(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let bias = make_f32_tensor(&[10.0, 20.0], vec![2]);
    let linear = Linear::from_weights(weight, Some(bias));

    let input = make_f32_tensor(&[3.0, 7.0], vec![1, 2]);
    let output = linear.forward(&input).unwrap();
    let data = output.as_slice_f32().unwrap();
    assert_f32_near(data, &[13.0, 27.0], 1e-5, "linear with bias");
}

#[test]
fn linear_without_bias_has_none() {
    let linear = Linear::new(4, 4, false);
    assert!(linear.bias.is_none(), "Linear(bias=false) should have no bias");
}

#[test]
fn linear_with_bias_has_some() {
    let linear = Linear::new(4, 4, true);
    assert!(linear.bias.is_some(), "Linear(bias=true) should have a bias");
    let bias = linear.bias.as_ref().unwrap();
    assert_eq!(bias.shape(), &[4]);
    // Should be zero-initialized
    let data = bias.as_slice_f32().unwrap();
    assert_f32_near(data, &[0.0, 0.0, 0.0, 0.0], 1e-10, "bias zero init");
}

#[test]
fn embedding_lookup_correctness() {
    // weight: 4 vocab, 3 dim
    // row 0: [10,20,30], row 1: [40,50,60], row 2: [70,80,90], row 3: [100,110,120]
    let weight = make_f32_tensor(
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
        vec![4, 3],
    );
    let emb = Embedding::from_weights(weight);

    // Lookup indices [0, 2, 1] as a [1, 3] tensor
    let indices = make_f32_tensor(&[0.0, 2.0, 1.0], vec![1, 3]);
    let output = emb.forward(&indices).unwrap();
    assert_eq!(output.shape(), &[1, 3, 3]);

    let data = output.as_slice_f32().unwrap();
    // Row 0 -> [10,20,30], Row 2 -> [70,80,90], Row 1 -> [40,50,60]
    assert_f32_near(
        data,
        &[10.0, 20.0, 30.0, 70.0, 80.0, 90.0, 40.0, 50.0, 60.0],
        1e-6,
        "embedding lookup",
    );
}

#[test]
fn embedding_out_of_bounds_errors() {
    let weight = make_f32_tensor(&[0.0; 12], vec![4, 3]);
    let emb = Embedding::from_weights(weight);

    // Index 4 == vocab_size, should error
    let indices = make_f32_tensor(&[4.0], vec![1, 1]);
    let result = emb.forward(&indices);
    assert!(result.is_err(), "Expected OOB error for index == vocab_size");
}

#[test]
fn embedding_out_of_bounds_large_index() {
    let weight = make_f32_tensor(&[0.0; 12], vec![4, 3]);
    let emb = Embedding::from_weights(weight);

    // Index 100 >> vocab_size=4
    let indices = make_f32_tensor(&[100.0], vec![1, 1]);
    let result = emb.forward(&indices);
    assert!(result.is_err(), "Expected OOB error for index >> vocab_size");
}

#[test]
fn layer_norm_output_mean_near_zero() {
    let input = make_f32_tensor(
        &[1.0, 5.0, 9.0, 13.0, 2.0, 4.0, 8.0, 16.0, 3.0, 6.0, 12.0, 24.0],
        vec![3, 4],
    );
    let ln = LayerNorm::new(vec![4], 1e-5);
    let output = ln.forward(&input).unwrap();
    let data = output.as_slice_f32().unwrap();

    for row in 0..3 {
        let start = row * 4;
        let row_data = &data[start..start + 4];
        let mean: f32 = row_data.iter().sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-4,
            "row {} mean should be ~0, got {}",
            row,
            mean
        );
    }
}

#[test]
fn layer_norm_output_variance_near_one() {
    let input = make_f32_tensor(
        &[1.0, 5.0, 9.0, 13.0, 2.0, 4.0, 8.0, 16.0, 3.0, 6.0, 12.0, 24.0],
        vec![3, 4],
    );
    let ln = LayerNorm::new(vec![4], 1e-5);
    let output = ln.forward(&input).unwrap();
    let data = output.as_slice_f32().unwrap();

    for row in 0..3 {
        let start = row * 4;
        let row_data = &data[start..start + 4];
        let mean: f32 = row_data.iter().sum::<f32>() / 4.0;
        let var: f32 = row_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(
            (var - 1.0).abs() < 0.1,
            "row {} variance should be ~1.0, got {}",
            row,
            var
        );
    }
}
