use super::*;
use crate::qtensor::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn permute_float() {
    let tensor = QTensor::<TestBackend, 1>::int8([
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
    ])
    .reshape([2, 2, 4]);

    let permuted = tensor.clone().permute([2, 1, 0]);

    let expected = TensorData::from([
        [[0., 8.], [4., 12.]],
        [[1., 9.], [5., 13.]],
        [[2., 10.], [6., 14.]],
        [[3., 11.], [7., 15.]],
    ]);

    permuted
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-1, 1e-1));

    // Test with negative axis
    let permuted = tensor.clone().permute([-1, 1, 0]);
    permuted
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(1e-1, 1e-1));

    // Test with the same axis
    let permuted = tensor.clone().permute([0, 1, 2]);
    permuted
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &tensor.dequantize().into_data(),
            Tolerance::rel_abs(1e-4, 1e-4), // dequant error should be the same
        );
}

#[test]
#[should_panic]
fn edge_repeated_axes() {
    let tensor = QTensor::<TestBackend, 1>::int8([
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
    ])
    .reshape([2, 2, 4]);

    // Test with a repeated axis
    let _ = tensor.permute([0, 0, 1]);
}

#[test]
#[should_panic]
fn edge_out_of_bound_axis() {
    let tensor = QTensor::<TestBackend, 1>::int8([
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
    ])
    .reshape([2, 2, 4]);

    // Test with an invalid axis
    let _ = tensor.permute([3, 0, 1]);
}
