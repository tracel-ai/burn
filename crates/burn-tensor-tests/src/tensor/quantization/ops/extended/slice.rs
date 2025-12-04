use super::*;
use crate::qtensor::*;
use burn_tensor::TensorData;
use burn_tensor::{Tolerance, s};

#[test]
fn should_support_full_sliceing_1d() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0]);
    let data = tensor.to_data();

    let output = tensor.slice([0..4]);

    output.into_data().assert_eq(&data, false);
}

#[test]
fn should_support_partial_sliceing_1d() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0]);

    let output = tensor.slice([1..3]);
    let expected = TensorData::from([1.0, 2.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_support_full_sliceing_2d() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data = tensor.to_data();

    let output = tensor.clone().slice([0..2]);
    output.into_data().assert_eq(&data, true);

    let output = tensor.slice([0..2, 0..3]);
    output.into_data().assert_eq(&data, true);
}

#[test]
fn should_support_partial_sliceing_2d() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.slice([0..2, 0..2]);
    let expected = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_support_partial_sliceing_3d() {
    let tensor = QTensor::<TestBackend, 3>::int8([
        [[0., 1., 2., 3.], [4., 5., 6., 7.]],
        [[8., 9., 10., 11.], [12., 13., 14., 15.]],
    ]);

    let output = tensor.slice([1..2, 1..2, 0..2]);
    let expected = TensorData::from([[[12.0, 13.0]]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_support_partial_sliceing_3d_non_contiguous() {
    let tensor = QTensor::<TestBackend, 3>::int8([
        [[0., 1., 2., 3.], [4., 5., 6., 7.]],
        [[8., 9., 10., 11.], [12., 13., 14., 15.]],
    ]);

    let output = tensor.transpose().slice([1..2, 1..2, 0..2]);
    let expected = TensorData::from([[[9.0, 13.0]]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_support_slice_assign_1d() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0]);
    let tensor_assigned = QTensor::<TestBackend, 1>::int8([10.0, 5.0]);

    let output = tensor.slice_assign([0..2], tensor_assigned);
    let expected = TensorData::from([10.0, 5.0, 2.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_slice_assign_2d() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_assigned = QTensor::<TestBackend, 2>::int8([[10.0, 5.0]]);

    let output = tensor.slice_assign([1..2, 0..2], tensor_assigned);
    let expected = TensorData::from([[0.0, 1.0, 2.0], [10.0, 5.0, 5.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn slice_should_not_corrupt_potentially_inplace_operations() {
    let tensor = QTensor::<TestBackend, 1>::int8([1.0, 2.0, 3.0, 4.0, 5.0]);
    let tensor = tensor.clone().slice([0..3]) + tensor.clone().slice([2..5]);

    let expected = TensorData::from([4., 6., 8.]);

    tensor
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn slice_assign_should_not_corrupt_potentially_inplace_operations() {
    let tensor = QTensor::<TestBackend, 1>::int8([1.0, 2.0, 3.0, 4.0, 5.0]);
    let values = QTensor::<TestBackend, 1>::int8([10., 20., 30.]);

    let tensor_1 = tensor.clone().slice_assign([0..3], values);
    let tensor_2 = tensor + 2;

    let expected = TensorData::from([10., 20., 30., 4., 5.]);

    tensor_1
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));

    let expected = TensorData::from([3., 4., 5., 6., 7.]);

    tensor_2
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn clamp_when_slice_exceeds_dimension() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0]);
    let data = tensor.to_data();

    let output = tensor.slice([0..4]);
    output.into_data().assert_eq(&data, true);
}

#[test]
fn negative_dimensions() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data = tensor.to_data();

    // Clamping to the tensor dimensions
    let output = tensor.clone().slice([0..4, 0..4]);
    output.into_data().assert_eq(&data, true);

    // Negative dimensions
    let output = tensor.clone().slice([0..1, 0..1]);
    let data = TensorData::from([[0.0f32]]);
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&data, Tolerance::rel_abs(2e-2, 1e-2));

    let output = tensor.slice(s![0..-1, 0..-2]);
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&data, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn missing_dimensions() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data = tensor.to_data();

    // Clamping to the tensor dimensions
    let output = tensor.clone().slice([0..4, 0..4]);
    output.into_data().assert_eq(&data, true);

    // Negative dimensions
    let data = TensorData::from([[0.0f32]]);
    let output = tensor.clone().slice(s![0..-1, 0..-2]);
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&data, Tolerance::rel_abs(2e-2, 1e-2));

    // Missing dimensions
    let output = tensor.clone().slice(s![0..1, ..]);
    let data = TensorData::from([[0.0f32, 1.0, 2.0]]);
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&data, Tolerance::rel_abs(2e-2, 1e-2));

    let output = tensor.clone().slice(s![.., 0..2]);
    let data = TensorData::from([[0.0f32, 1.0], [3.0, 4.0]]);
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&data, Tolerance::rel_abs(2e-2, 1e-2));

    let output = tensor.clone().slice([.., ..]);
    let data = TensorData::from([[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&data, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
#[should_panic]
fn should_panic_when_slice_with_too_many_dimensions() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0]);

    let _output = tensor.slice([0..1, 0..1]);
}
