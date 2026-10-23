use super::*;
use burn_tensor::{Shape, TensorData};

#[test]
fn should_support_rank() {
    let data = TensorData::from([0.0, 1.0, 2.0]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());
    assert_eq!(tensor.rank(), 1);

    let data = TensorData::from([[0.0, 1.0, 2.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());
    assert_eq!(tensor.rank(), 2);
}

#[test]
fn should_support_reshape_1d() {
    let data = TensorData::from([0.0, 1.0, 2.0]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    let output = tensor.clone().reshape([1, 3]);
    let expected = TensorData::from([[0.0, 1.0, 2.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_reshape_2d() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.clone().reshape([6]);
    let expected = TensorData::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_dim_infererence() {
    let data = TensorData::from([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0],
    ]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    // Infer the dimension via -1
    let reshaped = tensor.clone().reshape([2, -1]);
    assert_eq!(reshaped.shape(), [2, 6].into());

    // Infer the dimension via 0 (keep from the source) and -1 (infer)
    let reshaped = reshaped.reshape([0, 2, -1]);
    assert_eq!(reshaped.shape(), [2, 2, 3].into());

    // This is effectively as if we did a flatten
    let reshaped = tensor.clone().reshape([-1]);
    assert_eq!(reshaped.shape(), [12].into());

    // Keeping the first dimension the same (using 0)
    let reshaped = tensor.clone().reshape([0, 3]);
    assert_eq!(reshaped.shape(), [4, 3].into());
}

#[test]
fn should_reshape_to_shape_with_zero_length_dim() {
    let tensor = TestTensor::<2>::zeros([0, 2], &Default::default());

    // A `0` in a `Shape` is a zero-length dimension, not "keep the source dimension".
    let reshaped: TestTensor<2> = tensor.reshape(Shape::new([2, 0]));

    assert_eq!(reshaped.shape(), Shape::new([2, 0]));
    assert_eq!(reshaped.into_data().num_elements(), 0);
}

#[test]
#[should_panic(
    expected = "The given shape doesn't have the same number of elements as the current shape."
)]
fn should_panic_reshape_to_shape_with_different_num_elements() {
    let tensor = TestTensor::<1>::zeros([3], &Default::default());
    let _reshaped: TestTensor<2> = tensor.reshape(Shape::new([2, 2]));
}

#[test]
#[should_panic(
    expected = "The given shape doesn't have the same number of dimensions as the target tensor."
)]
fn should_panic_reshape_to_shape_with_wrong_rank() {
    let tensor = TestTensor::<1>::zeros([6], &Default::default());
    // Same element count (6), but the `Shape` has rank 2 while the target type has rank 3.
    let _reshaped: TestTensor<3> = tensor.reshape(Shape::new([2, 3]));
}

#[test]
fn should_not_corrupt_after_slice() {
    let zeros = TestTensor::<1>::zeros([2], &Default::default());
    zeros.clone().slice([1..2]).reshape([1]).exp();

    // May lead to zeroes being equal to [0.0, 1.0]
    zeros.into_data().assert_eq(
        &TestTensor::<1>::zeros([2], &Default::default()).to_data(),
        true,
    );
}

#[test]
#[should_panic(expected = "Repeated -1 in reshape")]
fn multiple_neg_ones() {
    let data = TensorData::from([0.0, 1.0, 2.0]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());
    let _data_actual = tensor.reshape([-1, -1]).into_data();
}

#[test]
#[should_panic(expected = "cannot contain negative dimensions")]
fn neg_value() {
    let data = TensorData::from([0.0, 1.0, 2.0]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());
    let _data_actual = tensor.reshape([-2, -1]).into_data();
}
