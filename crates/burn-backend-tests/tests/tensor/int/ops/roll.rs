use super::*;
use burn_tensor::TensorData;

#[test]
fn test_roll_empty() {
    let device = Default::default();
    let input = TestTensorInt::<2>::zeros([12, 0], &device);

    let result = input.clone().roll(&[1, 2], &[0, 1]);

    assert_eq!(&*result.shape(), &[12, 0]);

    // TODO: Rolling an empty tensor should return the same empty tensor;
    // but we have no way to compare tensor references yet.
}

#[test]
fn test_roll() {
    let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    // No-op shift:
    input
        .clone()
        .roll(&[0, 0], &[0, 1])
        .to_data()
        .assert_eq(&input.clone().to_data(), false);

    // Expected values match torch.roll(input, shifts=(1, -1), dims=(0, 1)).
    input
        .clone()
        .roll(&[1, -1], &[0, 1])
        .to_data()
        .assert_eq(&TensorData::from([[4, 5, 3], [1, 2, 0]]), false);

    input
        .clone()
        .roll(&[-1, 1], &[1, 0])
        .to_data()
        .assert_eq(&TensorData::from([[4, 5, 3], [1, 2, 0]]), false);

    input
        .clone()
        .roll(&[2 * 32 + 1, 3 * (-400) - 1], &[0, 1])
        .to_data()
        .assert_eq(&TensorData::from([[4, 5, 3], [1, 2, 0]]), false);
}

/// torch.roll(x, shifts, dims) shifts elements towards higher indices for a
/// positive shift, i.e. out[i] = x[(i - shift) % n]. Burn documents `roll`/`roll_dim`
/// as equivalent to `torch.roll` (see burn-book/src/building-blocks/tensor.md), so it
/// must match that direction rather than the opposite one.
#[test]
fn test_roll_matches_torch_direction() {
    let input = TestTensorInt::<1>::from([0, 1, 2, 3]);

    // torch.roll(torch.tensor([0, 1, 2, 3]), shifts=1) == tensor([3, 0, 1, 2])
    input
        .clone()
        .roll(&[1], &[0])
        .to_data()
        .assert_eq(&TensorData::from([3, 0, 1, 2]), false);

    // torch.roll(torch.tensor([0, 1, 2, 3]), shifts=1) == tensor([3, 0, 1, 2])
    input
        .clone()
        .roll_dim(1, 0)
        .to_data()
        .assert_eq(&TensorData::from([3, 0, 1, 2]), false);
}

#[should_panic]
#[test]
fn test_roll_dim_too_big() {
    let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    // Attempting to roll on a dimension that doesn't exist should panic
    let _d = input.roll(&[1], &[2]);
}

#[should_panic]
#[test]
fn test_roll_dim_too_small() {
    let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    // Attempting to roll on a dimension that doesn't exist should panic
    let _d = input.roll(&[1], &[-3]);
}

#[should_panic]
#[test]
fn test_roll_shift_size_mismatch() {
    let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    // Attempting to roll with a shift size that doesn't match the number of dimensions should panic
    let _d = input.roll(&[1, 2], &[0]);
}

#[test]
fn test_roll_dim() {
    let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    input
        .clone()
        .roll_dim(1, 0)
        .to_data()
        .assert_eq(&TensorData::from([[3, 4, 5], [0, 1, 2]]), false);

    // Expected values match torch.roll(input, shifts=-1, dims=1).
    input
        .clone()
        .roll_dim(-1, 1)
        .to_data()
        .assert_eq(&TensorData::from([[1, 2, 0], [4, 5, 3]]), false);
}

#[should_panic]
#[test]
fn test_roll_dim_dim_too_big() {
    let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    // Attempting to roll on a dimension that doesn't exist should panic
    let _d = input.roll_dim(1, 2);
}

#[should_panic]
#[test]
fn test_roll_dim_dim_too_small() {
    let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    // Attempting to roll on a dimension that doesn't exist should panic
    let _d = input.roll_dim(1, -3);
}
