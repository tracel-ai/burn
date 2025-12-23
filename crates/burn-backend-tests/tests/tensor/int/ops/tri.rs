use super::*;
use burn_tensor::TensorData;

#[test]
fn test_triu_negative_diagonal() {
    let tensor = TestTensorInt::<2>::from([[1, 1, 1], [1, 1, 1], [1, 1, 1]]);

    let output = tensor.triu(-1);
    let expected = TensorData::from([[1, 1, 1], [1, 1, 1], [0, 1, 1]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_triu_batch_tensors() {
    let tensor = TestTensorInt::<4>::from([
        [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
        [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
    ]);
    let output = tensor.triu(1);
    let expected = TensorData::from([
        [[[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]],
        [[[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn test_triu_too_few_dims() {
    let tensor = TestTensorInt::<1>::from([1, 2, 3]);
    let _output = tensor.triu(0);
}

#[test]
fn test_tril() {
    let tensor = TestTensor::<2>::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]);
    let output = tensor.tril(0);
    let expected = TensorData::from([[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_tril_positive_diagonal() {
    let tensor = TestTensorInt::<2>::from([[1, 1, 1], [1, 1, 1], [1, 1, 1]]);

    let output = tensor.tril(1);
    let expected = TensorData::from([[1, 1, 0], [1, 1, 1], [1, 1, 1]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_tril_negative_diagonal() {
    let tensor = TestTensorInt::<2>::from([[1, 1, 1], [1, 1, 1], [1, 1, 1]]);

    let output = tensor.tril(-1);
    let expected = TensorData::from([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_tril_batch_tensors() {
    let tensor = TestTensorInt::<4>::from([
        [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
        [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
    ]);
    let output = tensor.tril(1);
    let expected = TensorData::from([
        [[[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]],
        [[[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]],
    ]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn test_tril_too_few_dims() {
    let tensor = TestTensorInt::<1>::from([1, 2, 3]);
    let _output = tensor.tril(0);
}
