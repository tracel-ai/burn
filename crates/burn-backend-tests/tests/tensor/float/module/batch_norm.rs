use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::module::{batch_norm, batch_norm_train};

#[test]
fn test_batch_norm_forward() {
    let input = TestTensor::<3>::from([
        [[0.9601, 0.7277], [0.6272, 0.9034], [0.9378, 0.7230]],
        [[0.6356, 0.1362], [0.0249, 0.9509], [0.6600, 0.5945]],
    ]);
    let gamma = TestTensor::<1>::from([2.0, 3.0, 4.0]);
    let beta = TestTensor::<1>::from([0.5, -1.0, 2.0]);
    let mean = TestTensor::<1>::from([1.0, 2.0, 3.0]);
    let variance = TestTensor::<1>::from([3.0, 8.0, 15.0]);

    let output = batch_norm(input, gamma, beta, mean, variance, 1.0);

    let expected = TensorData::from([
        [[0.4601, 0.2277], [-2.3728, -2.0966], [-0.0622, -0.2770]],
        [[0.1356, -0.3638], [-2.9751, -2.0491], [-0.3400, -0.4055]],
    ]);
    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::relative(1e-5).set_half_precision_relative(5e-3),
    );
}

#[test]
fn test_batch_norm_train_normalizes_with_the_statistics_it_returns() {
    let input = TestTensor::<3>::from([
        [[0.9601, 0.7277], [0.6272, 0.9034], [0.9378, 0.7230]],
        [[0.6356, 0.1362], [0.0249, 0.9509], [0.6600, 0.5945]],
    ]);
    let gamma = TestTensor::<1>::from([2.0, 3.0, 4.0]);
    let beta = TestTensor::<1>::from([0.5, -1.0, 2.0]);
    let tolerance = Tolerance::relative(1e-4).set_half_precision_relative(5e-3);

    let (output, mean, variance) =
        batch_norm_train(input.clone(), gamma.clone(), beta.clone(), 1e-5);

    // The statistics of the batch, per channel, computed the long way.
    let expected_mean = input
        .clone()
        .swap_dims(0, 1)
        .reshape([3, 4])
        .mean_dim(1)
        .reshape([3]);
    let expected_variance = input
        .clone()
        .sub(expected_mean.clone().reshape([1, 3, 1]))
        .square()
        .swap_dims(0, 1)
        .reshape([3, 4])
        .mean_dim(1)
        .reshape([3]);
    let expected_output = batch_norm(
        input,
        gamma,
        beta,
        expected_mean.clone(),
        expected_variance.clone(),
        1e-5,
    );

    mean.into_data()
        .assert_approx_eq::<FloatElem>(&expected_mean.into_data(), tolerance);
    variance
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_variance.into_data(), tolerance);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_output.into_data(), tolerance);
}
