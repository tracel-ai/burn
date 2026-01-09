use super::*;
use burn_tensor::TensorData;

#[test]
fn expand_2d_bool() {
    let tensor = TestTensorBool::<1>::from([false, true, false]);
    let expanded_tensor = tensor.expand([3, 3]);

    let expected_data = TensorData::from([
        [false, true, false],
        [false, true, false],
        [false, true, false],
    ]);

    expanded_tensor.into_data().assert_eq(&expected_data, false);
}
