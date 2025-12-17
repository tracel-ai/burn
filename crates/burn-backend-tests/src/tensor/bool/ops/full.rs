use super::*;
use burn_tensor::TensorData;

#[test]
fn test_tensor_full() {
    let device = Default::default();
    let bool_tensor = TestTensorBool::<2>::full([2, 2], true, &device);
    bool_tensor
        .into_data()
        .assert_eq(&TensorData::from([[true, true], [true, true]]), false);

    let bool_tensor = TestTensorBool::<2>::full([2, 2], false, &device);
    bool_tensor
        .into_data()
        .assert_eq(&TensorData::from([[false, false], [false, false]]), false);
}
