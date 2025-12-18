use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_bool_mask_where_ops() {
    let device = Default::default();
    let tensor = TestTensorBool::<2>::from_data([[true, false], [false, false]], &device);
    let mask =
        TestTensorBool::<2>::from_bool(TensorData::from([[true, false], [false, true]]), &device);
    let value =
        TestTensorBool::<2>::from_data(TensorData::from([[false, true], [true, false]]), &device);

    let output = tensor.mask_where(mask, value);
    let expected = TensorData::from([[false, false], [false, false]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_bool_mask_fill_ops() {
    let device = Default::default();
    let tensor = TestTensorBool::<2>::from_data([[false, true], [false, false]], &device);
    let mask =
        TestTensorBool::<2>::from_bool(TensorData::from([[true, false], [false, true]]), &device);

    let output = tensor.mask_fill(mask, true);
    let expected = TensorData::from([[true, true], [false, true]]);

    output.into_data().assert_eq(&expected, false);
}
