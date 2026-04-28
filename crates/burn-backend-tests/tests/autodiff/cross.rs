use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn backward_basic() {
    let device = AutodiffDevice::new();
    let a = TestTensor::<2>::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &device,
    )
    .require_grad();
    let b = TestTensor::<2>::from_data(
        TensorData::from([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();

    // Simple cross product; grad is a vector of ones.
    let c = a.clone().cross(b.clone(), 1);
    let grads = c.backward();

    let a_grad = a.grad(&grads).unwrap().to_data();
    let b_grad = b.grad(&grads).unwrap().to_data();

    // For a: b×grad_out, where grad_out = [1,1,1]
    let expected_a = TensorData::from([[-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0]]);
    // For b: grad_out×a
    let expected_b = TensorData::from([[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]]);

    a_grad.assert_approx_eq::<FloatElem>(&expected_a, Tolerance::default());
    b_grad.assert_approx_eq::<FloatElem>(&expected_b, Tolerance::default());
}

#[test]
fn backward_after_sum() {
    let device = AutodiffDevice::new();
    let a = TestTensor::<2>::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &device,
    )
    .require_grad();
    let b = TestTensor::<2>::from_data(
        TensorData::from([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();

    // Sum reduces to scalar, but the gradient should be the same.
    let c = a.clone().cross(b.clone(), 1).sum();
    let grads = c.backward();

    let a_grad = a.grad(&grads).unwrap().to_data();
    let b_grad = b.grad(&grads).unwrap().to_data();

    let expected_a = TensorData::from([[-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0]]);
    let expected_b = TensorData::from([[1.0, -2.0, 1.0], [1.0, -2.0, 1.0]]);

    a_grad.assert_approx_eq::<FloatElem>(&expected_a, Tolerance::default());
    b_grad.assert_approx_eq::<FloatElem>(&expected_b, Tolerance::default());
}

#[test]
fn different_dim() {
    // Cross along a non-last dimension (dim 0) treats columns as 3-vectors.
    let device = AutodiffDevice::new();
    let a_raw = [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]];
    let b_raw = [[9.0, 6.0, 3.0], [8.0, 5.0, 2.0], [7.0, 4.0, 1.0]];

    let a = TestTensor::<2>::from_data(TensorData::from(a_raw), &device);
    let b = TestTensor::<2>::from_data(TensorData::from(b_raw), &device);
    let out = a.cross(b.clone(), 0);

    // Manually compute cross of each column vector using raw arrays
    let expected = [
        [
            a_raw[1][0] * b_raw[2][0] - a_raw[2][0] * b_raw[1][0],
            a_raw[1][1] * b_raw[2][1] - a_raw[2][1] * b_raw[1][1],
            a_raw[1][2] * b_raw[2][2] - a_raw[2][2] * b_raw[1][2],
        ],
        [
            a_raw[2][0] * b_raw[0][0] - a_raw[0][0] * b_raw[2][0],
            a_raw[2][1] * b_raw[0][1] - a_raw[0][1] * b_raw[2][1],
            a_raw[2][2] * b_raw[0][2] - a_raw[0][2] * b_raw[2][2],
        ],
        [
            a_raw[0][0] * b_raw[1][0] - a_raw[1][0] * b_raw[0][0],
            a_raw[0][1] * b_raw[1][1] - a_raw[1][1] * b_raw[0][1],
            a_raw[0][2] * b_raw[1][2] - a_raw[1][2] * b_raw[0][2],
        ],
    ];

    out.to_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from(expected), Tolerance::default());
}

#[test]
fn backward_non_last_dim() {
    // Backward through a cross on dim 0. The autodiff rule recurses through
    // float_cross with the same dim, so this also exercises the non-last-dim
    // forward path on the gradient pass.
    let device = AutodiffDevice::new();
    let a = TestTensor::<2>::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let b = TestTensor::<2>::from_data(
        TensorData::from([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
        &device,
    )
    .require_grad();

    // Cross on the column dimension, then permute and cross on the last dim.
    // Both paths share the same expected gradient magnitudes; we compare the
    // dim-0 output against the reference last-dim output to keep the check
    // backend-agnostic.
    let c0 = a.clone().cross(b.clone(), 0);
    let c_ref = a
        .clone()
        .permute([1, 0])
        .cross(b.clone().permute([1, 0]), 1)
        .permute([1, 0]);

    c0.to_data()
        .assert_approx_eq::<FloatElem>(&c_ref.to_data(), Tolerance::default());

    let grads = c0.sum().backward();
    assert!(a.grad(&grads).is_some());
    assert!(b.grad(&grads).is_some());
}
