mod common;

use common::{make_f32_tensor, assert_f32_near};

#[test]
fn reshape_basic_preserves_data() {
    let data: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    let t = make_f32_tensor(&data, vec![2, 3]);

    // [2,3] -> [3,2]
    let r1 = t.reshape(&[3, 2]).unwrap();
    assert_eq!(r1.shape(), &[3, 2]);
    let r1_data = r1.as_slice_f32().unwrap();
    assert_f32_near(r1_data, &data, 1e-6, "reshape [2,3]->[3,2]");

    // [3,2] -> [6]
    let r2 = r1.reshape(&[6]).unwrap();
    assert_eq!(r2.shape(), &[6]);
    let r2_data = r2.as_slice_f32().unwrap();
    assert_f32_near(r2_data, &data, 1e-6, "reshape [3,2]->[6]");
}

#[test]
fn reshape_non_contiguous_forces_copy() {
    // Create [[1,2,3],[4,5,6]], transpose to [[1,4],[2,5],[3,6]]
    let t = make_f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let transposed = t.transpose(0, 1).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);

    // Reshape non-contiguous tensor should force contiguous copy
    let flat = transposed.reshape(&[6]).unwrap();
    let data = flat.as_slice_f32().unwrap();
    // Transposed logical order: [1,4,2,5,3,6]
    assert_f32_near(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6, "reshape non-contiguous");
}

#[test]
fn reshape_element_count_mismatch_errors() {
    let t = make_f32_tensor(&[0.0; 6], vec![2, 3]);
    let result = t.reshape(&[4, 2]);
    assert!(result.is_err(), "Expected error for [2,3] -> [4,2] (6 != 8)");
}

#[test]
fn transpose_2d_swaps_elements() {
    // [[1,2,3],[4,5,6]] transposed -> [[1,4],[2,5],[3,6]]
    let t = make_f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let tr = t.transpose(0, 1).unwrap();
    assert_eq!(tr.shape(), &[3, 2]);

    // Make contiguous to check actual element order
    let c = tr.contiguous().unwrap();
    let data = c.as_slice_f32().unwrap();
    assert_f32_near(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6, "transpose 2d contiguous");
}

#[test]
fn transpose_3d_swaps_inner_dims() {
    // [2,3,4].transpose(1,2) -> [2,4,3]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = make_f32_tensor(&data, vec![2, 3, 4]);
    let tr = t.transpose(1, 2).unwrap();
    assert_eq!(tr.shape(), &[2, 4, 3]);

    // Make contiguous and verify an element
    let c = tr.contiguous().unwrap();
    let c_data = c.as_slice_f32().unwrap();

    // Original [0,1,2] = element at (b=0, row=1, col=2) = 1*4 + 2 = 6
    // After transpose(1,2): (b=0, row=2, col=1) -> should be 6
    // In contiguous layout: batch 0, row 2, col 1 = 0*12 + 2*3 + 1 = 7th element (index 7)
    assert_eq!(c_data[7], 6.0, "transposed element check");
}

#[test]
fn transpose_out_of_bounds_errors() {
    let t = make_f32_tensor(&[0.0; 6], vec![2, 3]);
    let result = t.transpose(0, 5);
    assert!(result.is_err(), "Expected error for transpose(0, 5) on 2D tensor");
}

#[test]
fn permute_3d_reorder() {
    // [2,3,4].permute([2,0,1]) -> [4,2,3]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = make_f32_tensor(&data, vec![2, 3, 4]);
    let p = t.permute(&[2, 0, 1]).unwrap();
    assert_eq!(p.shape(), &[4, 2, 3]);

    // Verify specific element: original(1, 2, 3) = 1*12 + 2*4 + 3 = 23
    // After permute [2,0,1]: new(3, 1, 2) should be 23
    let c = p.contiguous().unwrap();
    let c_data = c.as_slice_f32().unwrap();
    // Index in contiguous [4,2,3]: 3*6 + 1*3 + 2 = 23
    assert_eq!(c_data[23], 23.0, "permuted element check");
}

#[test]
fn permute_wrong_length_errors() {
    let t = make_f32_tensor(&[0.0; 24], vec![2, 3, 4]);
    let result = t.permute(&[1, 0]); // 2 dims for 3D tensor
    assert!(result.is_err(), "Expected error for permute with wrong number of dims");
}

#[test]
fn contiguous_strided_tensor_correct_copy() {
    // Create a non-contiguous tensor via transpose, then make contiguous
    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let t = make_f32_tensor(&data, vec![3, 4]);
    let tr = t.transpose(0, 1).unwrap();

    assert!(!tr.is_contiguous(), "Transposed tensor should not be contiguous");

    let c = tr.contiguous().unwrap();
    assert!(c.is_contiguous(), "After contiguous() should be contiguous");
    assert_eq!(c.shape(), &[4, 3]);

    // Original [3,4]: [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    // Transposed [4,3]: [[1,5,9],[2,6,10],[3,7,11],[4,8,12]]
    let c_data = c.as_slice_f32().unwrap();
    assert_f32_near(
        c_data,
        &[1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0],
        1e-6,
        "contiguous after transpose",
    );
}
