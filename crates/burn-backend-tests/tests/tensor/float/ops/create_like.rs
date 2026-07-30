use super::*;
use burn_tensor::Tolerance;
use burn_tensor::kind::Basic;
use burn_tensor::{Distribution, TensorData};

fn assert_like<const D: usize, K: Basic>(actual: &Tensor<D, K>, expected: &Tensor<D, K>) {
    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.device(), expected.device());
}

#[test]
fn should_support_empty_like() {
    let source = TestTensor::<3>::from_data(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    let created = source.empty_like();
    assert_like(&created, &source);
}

#[test]
fn should_support_zeros_like() {
    let source = TestTensor::<3>::from_data(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    let created = source.zeros_like();
    assert_like(&created, &source);

    let expected = TensorData::from([[[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]]]);

    created
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_ones_like() {
    let source = TestTensor::<3>::from_data(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    let created = source.ones_like();
    assert_like(&created, &source);

    let expected = TensorData::from([[[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]]]);

    created
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_randoms_like() {
    let source = TestTensor::<3>::from_data(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    let created = source.random_like(Distribution::Uniform(0.99999, 1.));
    assert_like(&created, &source);

    let expected = TensorData::from([[[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]]]);

    created
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
