use super::*;
use burn_tensor::TensorData;

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
