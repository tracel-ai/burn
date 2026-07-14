use super::*;
use burn_tensor::TensorData;

#[test]
fn test_triu() {
    let tensor = TestTensor::<2>::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]);
    let output = tensor.triu(0);
    let expected = TensorData::from([[1., 1., 1.], [0., 1., 1.], [0., 0., 1.]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_triu_positive_diagonal() {
    let tensor = TestTensor::<2>::from([[1, 1, 1], [1, 1, 1], [1, 1, 1]]);

    let output = tensor.triu(1);
    let expected = TensorData::from([[0, 1, 1], [0, 0, 1], [0, 0, 0]]);

    output.into_data().assert_eq(&expected, false);
}
