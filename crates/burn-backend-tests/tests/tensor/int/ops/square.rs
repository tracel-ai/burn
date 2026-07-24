use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_square_ops() {
    let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let tensor = TestTensorInt::<2>::from_data(data, &Default::default());

    let output = tensor.square();
    let expected = TensorData::from([[0, 1, 4], [9, 16, 25]]);

    output.into_data().assert_eq(&expected, false);
}
