use crate::*;
use burn_tensor::{TensorData, Tolerance};

#[cfg(feature = "std")]
use burn_tensor::might_panic;

#[test]
fn backward_basic() {
    let device = Default::default();
    let a = TestAutodiffTensor::<2>::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &device,
    )
    .require_grad();
    let b = TestAutodiffTensor::<2>::from_data(
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
    let device = Default::default();
    let a = TestAutodiffTensor::<2>::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &device,
    )
    .require_grad();
    let b = TestAutodiffTensor::<2>::from_data(
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

#[cfg(feature = "std")]
#[might_panic(reason = "not implemented: Cross product on non-last dimension")]
#[test]
fn different_dim() {
    // Also check when the cross is along a different dimension (e.g. dim 0).
    let device = Default::default();
    let a_raw = [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]];
    let b_raw = [[9.0, 6.0, 3.0], [8.0, 5.0, 2.0], [7.0, 4.0, 1.0]];

    let a = TestTensor::<2>::from_data(TensorData::from(a_raw), &device);
    let b = TestTensor::<2>::from_data(TensorData::from(b_raw), &device);
    // Cross along dim 0. Some backends (for example CubeCL) may not support
    // cross on non-last dimensions and will intentionally panic with a
    // message like "Cross product on non-last dimension not yet implemented".
    // In that case we treat the panic as a skipped test for that backend.
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
