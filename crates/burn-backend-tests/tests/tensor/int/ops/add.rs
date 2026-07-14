use super::*;
use burn_tensor::TensorData;

#[test]
fn test_add_d2_int() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);
    let tensor_2 = TestTensorInt::from([[6, 7, 8], [9, 10, 11]]);

    let output = tensor_1 + tensor_2;

    output
        .into_data()
        .assert_eq(&TensorData::from([[6, 8, 10], [12, 14, 16]]), false);
}

#[test]
fn test_add_broadcast_int() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2]]);
    let tensor_2 = TestTensorInt::from([[3, 4, 5], [6, 7, 8]]);

    let output = tensor_1 + tensor_2;

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 5, 7], [6, 8, 10]]), false);
}

#[test]
fn should_support_add_scalar_ops_int() {
    let scalar = 2;
    let tensor = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    let output = tensor + scalar;

    output
        .into_data()
        .assert_eq(&TensorData::from([[2, 3, 4], [5, 6, 7]]), false);
}

#[test]
fn scalar_add_not_contiguous() {
    let tensor = TestTensorInt::<1>::arange(0..32, &Default::default()).float();
    let tensor = tensor.reshape([1, 4, 4, 2]).permute([0, 3, 1, 2]);

    let tensor = tensor.slice([0..1, 0..2, 0..4, 0..4]);
    let before = tensor.clone();

    let after = tensor.add_scalar(0.0);

    before
        .into_data()
        .assert_approx_eq::<f32>(&after.into_data(), Default::default());
}

#[test]
fn scalar_add_not_contiguous_int() {
    let tensor = TestTensorInt::<1>::arange(0..32, &Default::default());
    let tensor = tensor.reshape([1, 4, 4, 2]).permute([0, 3, 1, 2]);

    let tensor = tensor.slice([0..1, 0..2, 0..4, 0..4]);
    let before = tensor.clone();

    let after = tensor.add_scalar(0);

    before.into_data().assert_eq(&after.into_data(), true);
}

#[test]
fn test_int_add_transposed() {
    // Non-contig on both sides via transpose.
    let a = TestTensorInt::<2>::from([[1, 2], [3, 4]]).transpose();
    let b = TestTensorInt::<2>::from([[10, 20], [30, 40]]).transpose();

    let output = a + b;

    output
        .into_data()
        .assert_eq(&TensorData::from([[11, 33], [22, 44]]), false);
}

#[test]
fn test_int_add_flipped() {
    // [1, 2, 3, 4] flipped + [10, 20, 30, 40] flipped = [44, 33, 22, 11]
    let a = TestTensorInt::<1>::from([1, 2, 3, 4]).flip([0]);
    let b = TestTensorInt::<1>::from([10, 20, 30, 40]).flip([0]);

    let output = a + b;

    output
        .into_data()
        .assert_eq(&TensorData::from([44, 33, 22, 11]), false);
}

#[test]
fn test_int_add_scalar_flipped() {
    // Scalar add on a flipped input. [1,2,3,4] flipped + 10 = [14,13,12,11]
    let a = TestTensorInt::<1>::from([1, 2, 3, 4]).flip([0]);

    let output = a + 10;

    output
        .into_data()
        .assert_eq(&TensorData::from([14, 13, 12, 11]), false);
}
