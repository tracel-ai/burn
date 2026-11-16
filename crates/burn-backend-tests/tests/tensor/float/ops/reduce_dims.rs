//! Multi-axis reductions preserve logical values across memory layouts.

use super::*;
use burn_tensor::{TensorData, Tolerance};

/// A `[2, 3, 4, 5]` tensor held channels-last in memory, and a contiguous
/// copy of the same values.
fn permuted_and_contiguous() -> (TestTensor<4>, TestTensor<4>) {
    let device = Default::default();
    let permuted = TestTensorInt::arange(0..(2 * 3 * 4 * 5), &device)
        .float()
        .reshape([2, 4, 5, 3])
        .permute([0, 3, 1, 2]);
    let contiguous = TestTensor::<4>::from_data(permuted.to_data(), &device);

    (permuted, contiguous)
}

fn one_dim_at_a_time(tensor: TestTensor<4>, dims: &[usize]) -> TestTensor<4> {
    dims.iter().fold(tensor, |tensor, &dim| tensor.sum_dim(dim))
}

fn assert_same(actual: TestTensor<4>, expected: TestTensor<4>) {
    let expected = expected.into_data();
    let actual = actual.into_data();

    assert_eq!(actual.shape(), expected.shape());
    actual.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn sum_dims_reduces_non_channel_dimensions_in_channels_last_memory() {
    let (permuted, contiguous) = permuted_and_contiguous();

    assert_same(
        permuted.sum_dims(&[0, 2, 3]),
        one_dim_at_a_time(contiguous, &[0, 2, 3]),
    );
}

#[test]
fn sum_dims_reduces_separated_axes() {
    let tensor = TestTensor::<3>::from_data(
        [
            [[-2.0, 1.0, 4.0], [3.0, -5.0, 4.0]],
            [[6.0, -1.0, 2.0], [-4.0, 3.0, 5.0]],
        ],
        &Default::default(),
    );

    tensor
        .sum_dims(&[0, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([[[10.0], [6.0]]]), Tolerance::default());
}

#[test]
fn sum_dims_preserves_permuted_broadcast_values() {
    let tensor =
        TestTensor::<3>::from_data([[[-2.0, 1.0, 4.0], [3.0, -5.0, 4.0]]], &Default::default())
            .expand([2, 2, 3])
            .permute([2, 1, 0]);

    tensor
        .sum_dims(&[0, 2])
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([[[6.0], [4.0]]]), Tolerance::default());
}

#[test]
fn dimensions_that_are_not_adjacent_in_any_order_still_sum_correctly() {
    let (permuted, contiguous) = permuted_and_contiguous();

    assert_same(
        permuted.sum_dims(&[0, 3]),
        one_dim_at_a_time(contiguous, &[0, 3]),
    );
}

#[test]
fn a_single_dimension_is_the_plain_sum_dim() {
    let (permuted, contiguous) = permuted_and_contiguous();

    assert_same(permuted.sum_dims(&[2]), contiguous.sum_dim(2));
}

#[test]
fn all_dimensions_at_once_is_the_total() {
    let (permuted, contiguous) = permuted_and_contiguous();

    let total = permuted.sum_dims(&[0, 1, 2, 3]);
    assert_eq!(total.dims(), [1, 1, 1, 1]);

    total
        .reshape([1])
        .into_data()
        .assert_approx_eq::<FloatElem>(&contiguous.sum().into_data(), Tolerance::default());
}

#[test]
fn a_zero_length_dimension_that_is_reduced_becomes_length_one_holding_the_identity() {
    let device = Default::default();
    let tensor = TestTensor::<2>::empty([0, 3], &device);

    let summed = tensor.sum_dims(&[0, 1]);
    assert_eq!(summed.dims(), [1, 1]);

    summed.into_data().assert_approx_eq::<FloatElem>(
        &TestTensor::<2>::from_data([[0.0]], &device).into_data(),
        Tolerance::default(),
    );
}

#[test]
fn a_zero_length_dimension_that_is_not_reduced_leaves_the_output_empty() {
    let tensor = TestTensor::<3>::empty([0, 2, 3], &Default::default());

    let summed = tensor.sum_dims(&[1, 2]);

    assert_eq!(summed.dims(), [0, 1, 1]);
    assert_eq!(summed.into_data().num_elements(), 0);
}

#[test]
fn a_singleton_sitting_between_two_folded_dimensions_does_not_break_the_run() {
    let device = Default::default();
    let contiguous = TestTensorInt::arange(0..(2 * 3 * 4), &device)
        .float()
        .reshape([2, 1, 3, 4]);

    assert_same(
        contiguous.clone().sum_dims(&[0, 2, 3]),
        one_dim_at_a_time(contiguous, &[0, 2, 3]),
    );
}

#[test]
fn a_folded_run_of_broadcast_dimensions_counts_each_value_once_per_repeat() {
    let device = Default::default();
    let broadcast = TestTensor::<3>::from_data([[[1.0, 2.0, 3.0]]], &device).expand([2, 4, 3]);

    let summed = broadcast.sum_dims(&[0, 1]);
    assert_eq!(summed.dims(), [1, 1, 3]);

    summed.into_data().assert_approx_eq::<FloatElem>(
        &TestTensor::<3>::from_data([[[8.0, 16.0, 24.0]]], &device).into_data(),
        Tolerance::default(),
    );
}
