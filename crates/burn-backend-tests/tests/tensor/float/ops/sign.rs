use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_sign_ops_float() {
    let tensor = TestTensor::<2>::from([[-0.2, -1.0, 2.0], [3.0, 0.0, -5.0]]);

    let output = tensor.sign();
    let expected = TensorData::from([[-1.0, -1.0, 1.0], [1.0, 0.0, -1.0]]);

    output.into_data().assert_eq(&expected, false);
}
