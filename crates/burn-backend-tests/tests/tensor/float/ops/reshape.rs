use super::*;
use burn_tensor::{Shape, TensorData, ZeroDimPlaceholders};

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

    let output = tensor.reshape([1, 3]);
    let expected = TensorData::from([[0.0, 1.0, 2.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_reshape_2d() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.reshape([6]);
    let expected = TensorData::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_resolve_reshape_dimensions() {
    let tensor = TestTensor::<2>::zeros([2, 6], &Default::default());

    let reshaped = tensor.clone().reshape([3, -1]);
    assert_eq!(reshaped.dims(), [3, 4]);

    let reshaped = tensor.reshape(ZeroDimPlaceholders([0, 2, -1]));
    assert_eq!(reshaped.dims(), [2, 2, 3]);
}

#[test]
fn should_reshape_empty_slice_to_higher_rank() {
    let tensor = TestTensor::<3>::zeros([1, 24, 8], &Default::default());
    let empty = tensor.slice([0..1, 0..24, 0..0]);

    // Arrays and concrete Shape targets agree, regardless of the array's integer type.
    for reshaped in [
        empty.clone().reshape([1i32, 24, 16, 0]),
        empty.clone().reshape([1usize, 24, 16, 0]),
        empty.reshape(Shape::new([1, 24, 16, 0])),
    ] {
        assert_eq!(reshaped.dims(), [1, 24, 16, 0]);
    }
}

#[test]
fn should_copy_zero_length_input_dimension() {
    let tensor = TestTensor::<2>::zeros([0, 2], &Default::default());
    let reshaped = tensor.reshape(ZeroDimPlaceholders([0usize, 0]));
    assert_eq!(reshaped.dims(), [0, 2]);
}

#[test]
fn should_infer_zero_length_dimension() {
    let tensor = TestTensor::<2>::zeros([0, 3], &Default::default());
    let reshaped = tensor.reshape([2, -1]);
    assert_eq!(reshaped.dims(), [2, 0]);
}

#[test]
#[should_panic(expected = "Zero placeholder at dimension 3 has no corresponding input dimension")]
fn should_reject_zero_placeholder_beyond_input_rank() {
    let tensor = TestTensor::<3>::zeros([1, 24, 0], &Default::default());
    let _ = tensor.reshape(ZeroDimPlaceholders([1, 24, 16, 0]));
}

#[test]
#[should_panic(expected = "doesn't have the same number of elements")]
fn should_reject_literal_zero_for_nonempty_tensor() {
    let tensor = TestTensor::<2>::zeros([2, 3], &Default::default());
    let _ = tensor.reshape([0, 3]);
}

#[test]
#[should_panic(expected = "Cannot infer a dimension when another resolved dimension is zero")]
fn should_reject_inference_with_literal_zero() {
    let tensor = TestTensor::<2>::zeros([0, 2], &Default::default());
    let _ = tensor.reshape([0, -1]);
}

#[test]
#[should_panic(expected = "Cannot infer a valid target shape")]
fn should_reject_nondivisible_inferred_dimension() {
    let tensor = TestTensor::<1>::zeros([5], &Default::default());
    let _ = tensor.reshape([2, -1]);
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
    let tensor = TestTensor::<1>::zeros([3], &Default::default());
    let _ = tensor.reshape([-1, -1]);
}

#[test]
#[should_panic(expected = "cannot contain negative dimensions")]
fn neg_value() {
    let tensor = TestTensor::<1>::zeros([3], &Default::default());
    let _ = tensor.reshape([-2, -1]);
}
