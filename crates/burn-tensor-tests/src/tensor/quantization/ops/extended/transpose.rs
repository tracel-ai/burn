use super::qtensor::*;
use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_transpose_ops() {
    let tensor = QTensor::<TestBackend, 1>::int8([
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
    ])
    .reshape([2, 2, 3]);

    let output = tensor.transpose();
    let expected = TensorData::from([
        [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
        [[6.0, 9.0], [7.0, 10.0], [8.0, 11.0]],
    ]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_swap_dims() {
    let tensor = QTensor::<TestBackend, 1>::int8([
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
    ])
    .reshape([2, 2, 3]);

    let output = tensor.swap_dims(0, 2);
    let expected = TensorData::from([
        [[0.0, 6.0], [3.0, 9.0]],
        [[1.0, 7.0], [4.0, 10.0]],
        [[2.0, 8.0], [5.0, 11.0]],
    ]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}
