//! The elementwise kernels on an operand that is not in logical dimension
//! order — a channels-last tensor presented in NCHW, as a convolution hands
//! one on. Each case runs the same operation on a contiguous copy and asks for
//! the same answer, so nothing here depends on a hand-computed expectation:
//! what is pinned is that the walk order a kernel picks for its operands never
//! changes what it computes.

use super::*;
use burn_tensor::Tolerance;

/// A `[2, 3, 4, 5]` tensor held channels-last in memory, and a contiguous copy
/// of the same values.
fn permuted_and_contiguous() -> (TestTensor<4>, TestTensor<4>) {
    let device = Default::default();
    let permuted = TestTensorInt::arange(0..(2 * 3 * 4 * 5), &device)
        .float()
        .reshape([2, 4, 5, 3])
        .permute([0, 3, 1, 2]);
    let contiguous = TestTensor::<4>::from_data(permuted.to_data(), &device);

    (permuted, contiguous)
}

fn assert_same(permuted: TestTensor<4>, contiguous: TestTensor<4>) {
    let expected = contiguous.into_data();
    let actual = permuted.into_data();

    assert_eq!(actual.shape, expected.shape);
    actual.assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn binary_with_both_operands_alive_matches_the_contiguous_answer() {
    let (permuted, contiguous) = permuted_and_contiguous();

    // Both operands stay alive, so the kernel writes a fresh output.
    assert_same(
        permuted.clone() * permuted.clone(),
        contiguous.clone() * contiguous.clone(),
    );
}

#[test]
fn binary_consuming_a_permuted_operand_matches_the_contiguous_answer() {
    let (left, contiguous) = permuted_and_contiguous();
    let (right, _) = permuted_and_contiguous();

    // Each operand owns its buffer and is consumed, so the kernel may write
    // over one of them in place; a clone would share the buffer and rule that
    // out.
    assert_same(left + right, contiguous.clone() + contiguous);
}

#[test]
fn binary_with_a_broadcast_operand_matches_the_contiguous_answer() {
    let (permuted, contiguous) = permuted_and_contiguous();
    let device = permuted.device();
    let per_channel = TestTensor::<4>::from_data([[[[1.5]], [[-2.0]], [[0.25]]]], &device);

    assert_same(permuted * per_channel.clone(), contiguous * per_channel);
}

#[test]
fn a_permuted_and_a_contiguous_operand_together_match_the_contiguous_answer() {
    let (permuted, contiguous) = permuted_and_contiguous();

    // The two operands disagree on their memory order; whichever walk the
    // kernel picks, one of them is read strided and the answer is the same.
    assert_same(
        permuted - contiguous.clone(),
        contiguous.clone() - contiguous,
    );
}

#[test]
fn scalar_on_a_permuted_operand_matches_the_contiguous_answer() {
    let (permuted, contiguous) = permuted_and_contiguous();

    assert_same(permuted.mul_scalar(2.5), contiguous.mul_scalar(2.5));
}

#[test]
fn unary_on_a_permuted_operand_matches_the_contiguous_answer() {
    let (permuted, contiguous) = permuted_and_contiguous();

    assert_same(permuted.exp(), contiguous.exp());
}
