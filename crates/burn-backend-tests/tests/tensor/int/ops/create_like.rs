use super::*;
use burn_tensor::TensorData;
use burn_tensor::kind::Basic;

fn assert_like<const D: usize, K: Basic>(actual: &Tensor<D, K>, expected: &Tensor<D, K>) {
    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.device(), expected.device());
}

#[test]
fn should_support_empty_like() {
    let source = TestTensorInt::<3>::from([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]);

    let created = source.zeros_like();
    assert_like(&created, &source);
}

#[test]
fn should_support_zeros_like() {
    let source = TestTensorInt::<3>::from([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]);

    let created = source.zeros_like();
    assert_like(&created, &source);

    let expected = TensorData::from([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]);

    created.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_ones_like() {
    let source = TestTensorInt::<3>::from([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]);

    let created = source.ones_like();
    assert_like(&created, &source);

    let expected = TensorData::from([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]]);

    created.into_data().assert_eq(&expected, false);
}
