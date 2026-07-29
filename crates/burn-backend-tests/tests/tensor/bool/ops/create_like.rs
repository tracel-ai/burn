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
    let source = TestTensorBool::<3>::from([
        [[false, true, false], [true, true, true]],
        [[false, false, false], [true, true, false]],
    ]);

    let created = source.empty_like();
    assert_like(&created, &source);
}

#[test]
fn should_support_zeros_like() {
    let source = TestTensorBool::<3>::from([
        [[false, true, false], [true, true, true]],
        [[false, false, false], [true, true, false]],
    ]);

    let created = source.zeros_like();
    assert_like(&created, &source);

    let expected = TensorData::from([
        [[false, false, false], [false, false, false]],
        [[false, false, false], [false, false, false]],
    ]);

    created.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_ones_like() {
    let source = TestTensorBool::<3>::from([
        [[false, true, false], [true, true, true]],
        [[false, false, false], [true, true, false]],
    ]);

    let created = source.ones_like();
    assert_like(&created, &source);

    let expected = TensorData::from([
        [[true, true, true], [true, true, true]],
        [[true, true, true], [true, true, true]],
    ]);

    created.into_data().assert_eq(&expected, false);
}
