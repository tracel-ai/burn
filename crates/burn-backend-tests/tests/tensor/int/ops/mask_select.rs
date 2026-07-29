use super::*;
use burn_tensor::TensorData;

#[test]
fn test_mask_select_int_1d() {
    let tensor = TestTensorInt::<1>::from([10, 20, 30, 40]);
    let mask = TestTensorBool::<1>::from([true, false, false, true]);

    let output = tensor.mask_select(mask);

    output
        .into_data()
        .assert_eq(&TensorData::from([10, 40]), false);
}

#[test]
fn test_mask_select_int_2d() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);
    let mask = TestTensorBool::<2>::from([[false, true, false], [true, false, true]]);

    let output = tensor.mask_select(mask);

    output
        .into_data()
        .assert_eq(&TensorData::from([2, 4, 6]), false);
}
