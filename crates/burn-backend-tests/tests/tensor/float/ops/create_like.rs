use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, TensorData};

#[test]
fn should_support_zeros_like() {
    let tensor = TestTensor::<3>::from_floats(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    let tensor = tensor.zeros_like();
    let expected = TensorData::from([[[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]]]);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_ones_like() {
    let tensor = TestTensor::<3>::from_floats(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    let tensor = tensor.ones_like();
    let expected = TensorData::from([[[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]]]);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_randoms_like() {
    let tensor = TestTensor::<3>::from_floats(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        &Default::default(),
    );

    let tensor = tensor.random_like(Distribution::Uniform(0.99999, 1.));
    let expected = TensorData::from([[[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]]]);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
