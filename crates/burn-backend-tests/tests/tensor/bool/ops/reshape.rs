use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_reshape_bool() {
    let data = TensorData::from([false, true, false]);
    let tensor = TestTensorBool::<1>::from_data(data, &Default::default());

    let output = tensor.clone().reshape([1, 3]);
    let expected = TensorData::from([[false, true, false]]);

    output.into_data().assert_eq(&expected, false);
}
