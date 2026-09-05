//! Reductions over several dimensions at once, on operands in and out of
//! logical dimension order. Each case asks for the same answer as reducing
//! the dimensions one at a time on a contiguous copy, so what is pinned is
//! that folding dimensions together never changes what is summed.

use super::*;
use burn_tensor::Tolerance;

/// A `[2, 3, 4, 5]` tensor held channels-last in memory, and a contiguous
/// copy of the same values.
fn permuted_and_contiguous() -> (TestTensor<4>, TestTensor<4>) {
    let device = Default::default();
    let permuted = TestTensorInt::arange(0..(2 * 3 * 4 * 5), &device)
        .float()
        .reshape([2, 4, 5, 3])
        .permute([0, 3, 1, 2]);
    permuted.device().sync().unwrap();

    let contiguous = TestTensor::<4>::from_data(permuted.to_data(), &device);
    contiguous.device().sync().unwrap();

    (permuted, contiguous)
}

fn one_dim_at_a_time(tensor: TestTensor<4>, dims: &[usize]) -> TestTensor<4> {
    dims.iter().fold(tensor, |tensor, &dim| tensor.sum_dim(dim))
}

fn assert_same(actual: TestTensor<4>, expected: TestTensor<4>) {
    let expected = expected.into_data();
    let actual = actual.into_data();

    assert_eq!(actual.shape, expected.shape);
    actual.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn every_non_channel_dimension_of_a_channels_last_tensor_folds_into_one() {
    // Batch, height and width sit next to each other in channels-last memory,
    // so this is the case that reduces in a single launch.
    let (permuted, contiguous) = permuted_and_contiguous();

    assert_same(
        permuted.sum_dims(&[0, 2, 3]),
        one_dim_at_a_time(contiguous, &[0, 2, 3]),
    );
}

#[test]
fn every_non_channel_dimension_of_a_contiguous_tensor_is_reduced_in_two() {
    // In logical order the batch dimension is separated from height and width
    // by the channels, so the innermost pair reduces first and the batch after.
    let (_, contiguous) = permuted_and_contiguous();

    assert_same(
        contiguous.clone().sum_dims(&[0, 2, 3]),
        one_dim_at_a_time(contiguous, &[0, 2, 3]),
    );
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
