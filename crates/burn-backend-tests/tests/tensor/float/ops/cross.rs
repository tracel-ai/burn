use super::*;
use burn_tensor::TensorData;

#[test]
fn test_cross_3d_last_dim() {
    let tensor_1 = TestTensor::<2>::from([[1.0, 3.0, -5.0], [2.0, -1.0, 4.0]]);
    let tensor_2 = TestTensor::from([[4.0, -2.0, 1.0], [3.0, 5.0, -2.0]]);

    let output = tensor_1.cross(tensor_2, -1);

    output.into_data().assert_eq(
        &TensorData::from([[-7.0, -21.0, -14.0], [-18.0, 16.0, 13.0]]),
        false,
    );
}

#[test]
fn test_cross_3d_non_contiguous_last_dim() {
    let tensor_1 = TestTensor::<2>::from([[1.0, 3.0, -5.0], [2.0, -1.0, 4.0]]);
    let tensor_2 = TestTensor::from([[4.0, 3.0], [-2.0, 5.0], [1.0, -2.0]]);

    let output = tensor_1.cross(tensor_2.permute([1, 0]), -1);

    output.into_data().assert_eq(
        &TensorData::from([[-7.0, -21.0, -14.0], [-18.0, 16.0, 13.0]]),
        false,
    );
}

#[test]
fn test_cross_3d_dim0() {
    let tensor_1 = TestTensor::<2>::from([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);
    let tensor_2 = TestTensor::from([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]);

    let output = tensor_1.cross(tensor_2, 0);

    output.into_data().assert_eq(
        &TensorData::from([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
        false,
    );
}

#[test]
fn test_cross_4d_middle_dim() {
    // Shape [1, 3, 2]; cross along dim=1 should match the corresponding
    // permuted last-dim cross.
    let tensor_1 = TestTensor::<3>::from([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]);
    let tensor_2 = TestTensor::from([[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]]);

    let output = tensor_1.cross(tensor_2, 1);

    output.into_data().assert_eq(
        &TensorData::from([[[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]]),
        false,
    );
}

#[test]
fn test_cross_non_last_dim_broadcast() {
    // Broadcast on a non-last cross dim: lhs shape [3, 1] vs rhs shape [3, 4]
    // crossed along dim=0 must equal the same op on the permuted [1, 3] vs
    // [4, 3] along dim=1 (last) and then permuted back.
    let lhs = TestTensor::<2>::from([[1.0], [2.0], [3.0]]);
    let rhs = TestTensor::<2>::from([
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0],
        [12.0, 13.0, 14.0, 15.0],
    ]);

    let non_last = lhs.clone().cross(rhs.clone(), 0);
    let last_dim = lhs
        .permute([1, 0])
        .cross(rhs.permute([1, 0]), 1)
        .permute([1, 0]);

    non_last.into_data().assert_eq(&last_dim.into_data(), false);
}

#[test]
fn test_cross_non_last_dim_matches_permuted_last_dim() {
    // Cross on a non-last dim must equal: permute -> cross on last dim ->
    // permute back. Here we cross [N, 3] along dim=1 (last) and along dim=0
    // after a transpose of the same data.
    let a = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b = TestTensor::<2>::from([[7.0, 8.0, 9.0], [1.0, 0.0, -1.0]]);

    let last_dim = a.clone().cross(b.clone(), 1);
    let non_last = a.permute([1, 0]).cross(b.permute([1, 0]), 0);

    last_dim
        .into_data()
        .assert_eq(&non_last.permute([1, 0]).into_data(), false);
}

#[test]
fn test_cross_3d_broadcast() {
    let tensor_1 = TestTensor::<2>::from([[1.0, 3.0, -5.0]]);
    let tensor_2 = TestTensor::from([[4.0, -2.0, 1.0], [3.0, 5.0, -2.0]]);

    let output = tensor_1.cross(tensor_2, -1);

    output.into_data().assert_eq(
        &TensorData::from([[-7.0, -21.0, -14.0], [19.0, -13.0, -4.0]]),
        false,
    );
}

#[test]
fn test_cross_4d_last_dim() {
    let tensor_1 = TestTensor::<3>::from([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
    let tensor_2 = TestTensor::from([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]);

    let output = tensor_1.cross(tensor_2, -1);

    output.into_data().assert_eq(
        &TensorData::from([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]),
        false,
    );
}

// Helper to compute expected cross product for 2-D (N × 3) tensors.
fn manual_cross(a: &[[f32; 3]], b: &[[f32; 3]]) -> Vec<[f32; 3]> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            [
                x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0],
            ]
        })
        .collect()
}

#[test]
fn forward_matches_manual_cross() {
    let a_raw = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b_raw = [[7.0, 8.0, 9.0], [1.0, 0.0, -1.0]];
    let a = TestTensor::<2>::from(a_raw);
    let b = TestTensor::<2>::from(b_raw);

    let out = a.cross(b.clone(), 1);
    let expected_vec = manual_cross(&a_raw, &b_raw);
    let expected: [[f32; 3]; 2] = [expected_vec[0], expected_vec[1]];

    out.into_data()
        .assert_eq(&TensorData::from(expected), false);
}
