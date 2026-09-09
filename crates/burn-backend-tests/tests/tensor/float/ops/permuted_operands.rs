//! The elementwise kernels on an operand that is not in logical dimension
//! order — a channels-last tensor presented in NCHW, as a convolution hands
//! one on. Each case runs the same operation on a contiguous copy and asks for
//! the same answer, so nothing here depends on a hand-computed expectation:
//! what is pinned is that the walk order a kernel picks for its operands never
//! changes what it computes.

use super::*;
use burn_tensor::{TensorData, Tolerance};

/// A `[2, 3, 4, 5]` tensor held channels-last in memory, and a contiguous copy
/// of the same values.
fn permuted_and_contiguous() -> (TestTensor<4>, TestTensor<4>) {
    let device = Default::default();
    let permuted = TestTensorInt::arange(0..(2 * 3 * 4 * 5), &device)
        .float()
        .div_scalar(120.0)
        .reshape([2, 4, 5, 3])
        .permute([0, 3, 1, 2]);
    let contiguous = TestTensor::<4>::from_data(permuted.to_data(), &device);

    (permuted, contiguous)
}

fn assert_same<const D: usize>(permuted: TestTensor<D>, contiguous: TestTensor<D>) {
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

#[test]
fn expanded_singleton_keeps_the_contiguous_answer() {
    let device = Default::default();
    let tensor = TestTensorInt::arange(0..1024, &device).float();
    let expanded = tensor.expand([1, 1024]);
    let contiguous = TestTensor::<2>::from_data(expanded.to_data(), &device);
    assert_same(
        expanded.clone().mul_scalar(2.5),
        contiguous.clone().mul_scalar(2.5),
    );
    assert_same(expanded.clone().abs(), contiguous.clone().abs());
    assert_same(expanded.clone() + expanded, contiguous.clone() + contiguous);
}

#[test]
fn two_different_nonlogical_layouts_match_in_both_operand_orders() {
    let device = Default::default();
    let values = TestTensorInt::arange(0..24, &device).float();
    let left = values.clone().reshape([2, 4, 3]).permute([0, 2, 1]);
    let right = values.reshape([3, 2, 4]).permute([1, 0, 2]);
    let left_contiguous = TestTensor::<3>::from_data(left.to_data(), &device);
    let right_contiguous = TestTensor::<3>::from_data(right.to_data(), &device);
    assert_same(
        left.clone() - right.clone(),
        left_contiguous.clone() - right_contiguous.clone(),
    );
    assert_same(right - left, right_contiguous - left_contiguous);
}

#[test]
fn padded_channels_last_operands_match_for_each_launcher() {
    let device = Default::default();
    // Keep eight of twelve channels: vectorizable rows with padding between them.
    let tensor = TestTensorInt::arange(0..(2 * 4 * 5 * 12), &device)
        .float()
        .div_scalar(480.0)
        .reshape([2, 4, 5, 12])
        .slice([0..2, 0..4, 0..5, 0..8])
        .permute([0, 3, 1, 2]);
    let contiguous = TestTensor::<4>::from_data(tensor.to_data(), &device);
    assert_same(
        tensor.clone() * tensor.clone(),
        contiguous.clone() * contiguous.clone(),
    );
    assert_same(
        tensor.clone().powf(tensor.clone()),
        contiguous.clone().powf(contiguous.clone()),
    );
    assert_same(
        tensor.clone().mul_scalar(2.5),
        contiguous.clone().mul_scalar(2.5),
    );
    assert_same(tensor.clone().exp(), contiguous.clone().exp());
    assert_same(tensor.abs(), contiguous.abs());
}

#[test]
fn large_spatial_broadcast_matches_the_contiguous_answer() {
    let device = Default::default();
    let tensor = TestTensorInt::arange(0..(2 * 8 * 16 * 16), &device)
        .float()
        .reshape([2, 16, 16, 8])
        .permute([0, 3, 1, 2]);
    let spatial = TestTensorInt::arange(0..(16 * 16), &device)
        .float()
        .reshape([1, 1, 16, 16]);
    let contiguous = TestTensor::<4>::from_data(tensor.to_data(), &device);
    assert_same(tensor * spatial.clone(), contiguous * spatial);
}

#[test]
fn fresh_elementwise_outputs_can_be_flattened_in_logical_order() {
    let (permuted, contiguous) = permuted_and_contiguous();
    // Retain both inputs so every operation must allocate an output. Flattening
    // must preserve NCHW value order, independent of the input's physical layout.
    assert_same(
        permuted.clone().mul_scalar(2.5).reshape([120]),
        contiguous.clone().mul_scalar(2.5).reshape([120]),
    );
    assert_same(
        (permuted.clone() * permuted.clone()).reshape([2, 60]),
        (contiguous.clone() * contiguous.clone()).reshape([2, 60]),
    );
    assert_same(
        permuted.clone().exp().reshape([120]),
        contiguous.clone().exp().reshape([120]),
    );
    assert_same(
        permuted.clone().abs().reshape([120]),
        contiguous.clone().abs().reshape([120]),
    );
    assert_same(
        permuted.clone().atan2(permuted.clone()).reshape([120]),
        contiguous.clone().atan2(contiguous.clone()).reshape([120]),
    );
}

#[test]
fn binary_reusing_the_right_operand_preserves_value_order() {
    let (left, left_contiguous) = permuted_and_contiguous();
    let (right, right_contiguous) = permuted_and_contiguous();
    // Only the right input may be overwritten. Subtraction checks operand order
    // as well as the inverse permutation applied to the reused result.
    let right = right.mul_scalar(2.0);
    let right_contiguous = right_contiguous.mul_scalar(2.0);
    assert_same(
        (left.clone() - right).reshape([120]),
        (left_contiguous.clone() - right_contiguous).reshape([120]),
    );
}

#[test]
fn empty_binary_outputs_bypass_buffer_reuse_checks() {
    let device = Default::default();
    let empty = TestTensor::<2>::from_data(TensorData::new(Vec::<f32>::new(), [0, 8]), &device);
    let row = TestTensor::<2>::ones([1, 8], &device);
    assert_eq!(
        (empty.clone() + row.clone()).into_data().shape.as_slice(),
        &[0, 8]
    );
    assert_eq!(empty.atan2(row).into_data().shape.as_slice(), &[0, 8]);
}
