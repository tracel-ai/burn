use super::*;
use burn_tensor::TensorData;

#[ignore = "0 size resources are not yet supported"]
#[test]
fn test_roll_empty() {
    let device = Default::default();
    let input = TestTensorInt::<2>::zeros([12, 0], &device);

    let result = input.clone().roll(&[1, 2], &[0, 1]);

    assert_eq!(result.shape().dims, &[12, 0]);

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

    input
        .clone()
        .roll(&[1, -1], &[0, 1])
        .to_data()
        .assert_eq(&TensorData::from([[5, 3, 4], [2, 0, 1]]), false);

    input
        .clone()
        .roll(&[-1, 1], &[1, 0])
        .to_data()
        .assert_eq(&TensorData::from([[5, 3, 4], [2, 0, 1]]), false);

    input
        .clone()
        .roll(&[2 * 32 + 1, 3 * (-400) - 1], &[0, 1])
        .to_data()
        .assert_eq(&TensorData::from([[5, 3, 4], [2, 0, 1]]), false);
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

    input
        .clone()
        .roll_dim(-1, 1)
        .to_data()
        .assert_eq(&TensorData::from([[2, 0, 1], [5, 3, 4]]), false);
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
