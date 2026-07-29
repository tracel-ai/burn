use super::*;
use burn_tensor::TensorData;

#[test]
fn test_mask_select_bool() {
    // Bool-kind sources go through the dedicated bool select dispatch path.
    let tensor = TestTensorBool::<1>::from([true, false, true]);
    let mask = TestTensorBool::<1>::from([true, true, false]);

    let output = tensor.mask_select(mask);

    output
        .into_data()
        .assert_eq(&TensorData::from([true, false]), false);
}
