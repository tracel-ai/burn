use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_mask_fill_swap_dims() {
    let device = Default::default();
    let tensor_1 = TestTensorInt::arange(0..16, &device).float();
    let tensor_1 = tensor_1.reshape([2, 2, 4]);
    let tensor_1 = tensor_1.swap_dims(0, 2);

    let mask = tensor_1.clone().lower_equal_elem(5.0);
    let output = tensor_1.clone().mask_fill(mask, -5.0);

    let expected = TensorData::from([
        [[-5.0, 8.0], [-5.0, 12.0]],
        [[-5.0, 9.0], [-5.0, 13.0]],
        [[-5.0, 10.0], [6.0, 14.0]],
        [[-5.0, 11.0], [7.0, 15.0]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_mask_where_ops() {
    let device = Default::default();
    let tensor = TestTensor::from_data([[1.0, 7.0], [2.0, 3.0]], &device);
    let mask =
        TestTensorBool::<2>::from_bool(TensorData::from([[true, false], [false, true]]), &device);
    let value = TestTensor::<2>::from_data(TensorData::from([[1.8, 2.8], [3.8, 4.8]]), &device);

    let output = tensor.mask_where(mask, value);
    let expected = TensorData::from([[1.8, 7.0], [2.0, 4.8]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_mask_where_broadcast_int() {
    let device = Default::default();
    // When broadcasted, the input [[2, 3], [4, 5]] is repeated 4 times
    let tensor = TestTensorInt::<1>::arange(2..6, &device).reshape([1, 2, 2]);
    let mask = TestTensorBool::<3>::from_bool(
        TensorData::from([
            [[true, false], [false, true]],
            [[false, true], [true, false]],
            [[false, false], [false, false]],
            [[true, true], [true, true]],
        ]),
        &device,
    );
    let value = TestTensorInt::<3>::ones([4, 2, 2], &device);

    let output = tensor.mask_where(mask, value);
    let expected = TensorData::from([
        [[1, 3], [4, 1]],
        [[2, 1], [1, 5]],
        [[2, 3], [4, 5]],
        [[1, 1], [1, 1]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_mask_where_broadcast() {
    let device = Default::default();
    // When broadcasted, the input [[2, 3], [4, 5]] is repeated 4 times
    let tensor = TestTensorInt::<1>::arange(2..6, &device).reshape([1, 2, 2]);
    let mask = TestTensorBool::<3>::from_bool(
        TensorData::from([
            [[true, false], [false, true]],
            [[false, true], [true, false]],
            [[false, false], [false, false]],
            [[true, true], [true, true]],
        ]),
        &device,
    );
    let value = TestTensor::<3>::ones([4, 2, 2], &device);

    let output = tensor.float().mask_where(mask, value);
    let expected = TensorData::from([
        [[1., 3.], [4., 1.]],
        [[2., 1.], [1., 5.]],
        [[2., 3.], [4., 5.]],
        [[1., 1.], [1., 1.]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_mask_where_broadcast_value_small() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(2..4, &device);
    let mask = TestTensorBool::<1>::from_bool(TensorData::from([true, false]), &device);
    let value = TestTensor::<1>::ones([1], &device);

    let output = tensor.float().mask_where(mask, value);
    let expected = TensorData::from([1., 3.]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_handle_mask_where_nans() {
    let device = Default::default();
    let tensor = TestTensor::from_data(
        [
            [f32::NAN, f32::NAN, f32::NAN],
            [f32::NAN, f32::NAN, f32::NAN],
            [f32::NAN, f32::NAN, f32::NAN],
        ],
        &device,
    );
    let mask = TestTensorBool::<2>::from_bool(
        TensorData::from([
            [true, true, true],
            [true, true, false],
            [false, false, false],
        ]),
        &device,
    );
    let value = TestTensor::<2>::from_data(
        TensorData::from([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]]),
        &device,
    );

    let output = tensor.mask_where(mask, value);
    let expected = TensorData::from([
        [0.9, 0.8, 0.7],
        [0.6, 0.5, f32::NAN],
        [f32::NAN, f32::NAN, f32::NAN],
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_mask_fill_ops() {
    let device = Default::default();
    let tensor = TestTensor::from_data([[1.0, 7.0], [2.0, 3.0]], &device);
    let mask =
        TestTensorBool::<2>::from_bool(TensorData::from([[true, false], [false, true]]), &device);

    let output = tensor.mask_fill(mask, 2.0);
    let expected = TensorData::from([[2.0, 7.0], [2.0, 2.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_mask_fill_broadcasted() {
    let device = Default::default();
    let tensor = TestTensor::zeros([1, 4, 2, 2], &device);
    let mask = TestTensorBool::<4>::from_bool(
        TensorData::from([[[[true, false], [false, true]]]]),
        &device,
    );

    let output = tensor.mask_fill(mask, 2.0);
    let expected = TensorData::from([[
        [[2., 0.], [0., 2.]],
        [[2., 0.], [0., 2.]],
        [[2., 0.], [0., 2.]],
        [[2., 0.], [0., 2.]],
    ]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_int_mask_where_ops() {
    let device = Default::default();
    let tensor = TestTensorInt::<2>::from_data([[1, 7], [2, 3]], &device);
    let mask =
        TestTensorBool::<2>::from_bool(TensorData::from([[true, false], [false, true]]), &device);
    let value = TestTensorInt::<2>::from_data(TensorData::from([[8, 9], [10, 11]]), &device);

    let output = tensor.mask_where(mask, value);
    let expected = TensorData::from([[8, 7], [2, 11]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_int_mask_fill_ops() {
    let device = Default::default();
    let tensor = TestTensorInt::<2>::from_data([[1, 7], [2, 3]], &device);
    let mask =
        TestTensorBool::<2>::from_bool(TensorData::from([[true, false], [false, true]]), &device);

    let output = tensor.mask_fill(mask, 9);
    let expected = TensorData::from([[9, 7], [2, 9]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn float_mask_fill_infinite() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [f32::NEG_INFINITY, f32::NEG_INFINITY],
            [f32::NEG_INFINITY, f32::NEG_INFINITY],
        ],
        &device,
    );
    let mask =
        TestTensorBool::<2>::from_bool(TensorData::from([[true, false], [false, true]]), &device);

    let output = tensor.mask_fill(mask, 10.0f32);
    let expected = TensorData::from([[10f32, f32::NEG_INFINITY], [f32::NEG_INFINITY, 10f32]]);

    output.into_data().assert_eq(&expected, false);
}
