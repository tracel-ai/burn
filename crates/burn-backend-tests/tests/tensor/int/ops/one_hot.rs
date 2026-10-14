use super::*;
use burn_tensor::TensorData;

#[test]
fn int_should_support_one_hot() {
    let tensor = TestTensorInt::<1>::from([0, 1, 4]);
    let one_hot_tensor: TestTensorInt<2> = tensor.one_hot(5);
    let expected = TensorData::from([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]]);
    one_hot_tensor.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic]
fn int_one_hot_should_panic_when_index_exceeds_number_of_classes() {
    let tensor = TestTensorInt::<1>::from([5]);
    let _result: TestTensorInt<2> = tensor.one_hot(5);
}

#[test]
#[should_panic]
fn int_one_hot_should_panic_when_number_of_classes_is_zero() {
    let tensor = TestTensorInt::<1>::from([2]);
    let _result: TestTensorInt<2> = tensor.one_hot(0);
}

#[test]
fn one_hot_fill_with_positive_axis_and_indices() {
    let tensor = TestTensorInt::<2>::from([[1, 9], [2, 4]]);
    let expected = TensorData::from([
        [
            [1, 1],
            [3, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 3],
        ],
        [
            [1, 1],
            [1, 1],
            [3, 1],
            [1, 1],
            [1, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ],
    ]);

    let one_hot_tensor: TestTensorInt<3> = tensor.one_hot_fill(10, 3.0, 1.0, 1);

    one_hot_tensor.into_data().assert_eq(&expected, false);
}

// I64 is not supported on every backend (e.g. WGSL has no i64)
#[cfg(feature = "flex")]
#[test]
fn one_hot_fill_should_keep_dtype() {
    let tensor = TestTensorInt::<1>::from([0, 2]).cast(burn_tensor::DType::I64);

    let one_hot_tensor: TestTensorInt<2> = tensor.one_hot_fill(3, 5.0, 1.0, -1);

    assert_eq!(one_hot_tensor.dtype(), burn_tensor::DType::I64);
    one_hot_tensor
        .into_data()
        .assert_eq(&TensorData::from([[5i64, 1, 1], [1, 1, 5]]), false);
}

// U32 is not supported on every backend
#[cfg(feature = "flex")]
#[test]
fn one_hot_fill_unsigned_with_on_value_below_off_value() {
    // `on_value - off_value` would underflow in an unsigned dtype
    let tensor = TestTensorInt::<1>::from([0, 2]).cast(burn_tensor::DType::U32);

    let one_hot_tensor: TestTensorInt<2> = tensor.one_hot_fill(3, 0.0, 1.0, -1);

    assert_eq!(one_hot_tensor.dtype(), burn_tensor::DType::U32);
    one_hot_tensor
        .into_data()
        .assert_eq(&TensorData::from([[0u32, 1, 1], [1, 1, 0]]), false);
}

#[test]
fn one_hot_fill_out_of_range_indices_give_off_value_rows() {
    // Valid indices are in [-num_classes, num_classes); -3 and 2 are the bounds.
    let tensor = TestTensorInt::<1>::from([0, 7, -4, 2, -3]);

    let one_hot_tensor: TestTensorInt<2> = tensor.one_hot_fill(3, 5.0, 1.0, -1);

    let expected = TensorData::from([[5, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 5], [5, 1, 1]]);
    one_hot_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn one_hot_fill_out_of_range_indices_with_axis_0() {
    let tensor = TestTensorInt::<1>::from([1, 3, -1]);

    let one_hot_tensor: TestTensorInt<2> = tensor.one_hot_fill(3, 5.0, 1.0, 0);

    let expected = TensorData::from([[1, 1, 1], [5, 1, 1], [1, 1, 5]]);
    one_hot_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn one_hot_fill_out_of_range_indices_with_middle_axis() {
    let tensor = TestTensorInt::<2>::from([[1, 5], [-4, 0]]);

    let one_hot_tensor: TestTensorInt<3> = tensor.one_hot_fill(3, 5.0, 1.0, 1);

    let expected = TensorData::from([[[1, 1], [5, 1], [1, 1]], [[1, 5], [1, 1], [1, 1]]]);
    one_hot_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn one_hot_fill_zero_classes_gives_empty_axis() {
    let tensor = TestTensorInt::<1>::from([0, 1, -1]);

    let one_hot_tensor: TestTensorInt<2> = tensor.one_hot_fill(0, 5.0, 1.0, -1);

    assert_eq!(one_hot_tensor.dims(), [3, 0]);
}

#[test]
#[should_panic(expected = "containing indexes outside [-5, 5)")]
fn int_one_hot_should_panic_when_index_below_minus_number_of_classes() {
    let tensor = TestTensorInt::<1>::from([-6]);
    let _result: TestTensorInt<2> = tensor.one_hot(5);
}

// U32 is not supported on every backend
#[cfg(feature = "flex")]
#[test]
fn int_one_hot_unsigned_indices() {
    let tensor = TestTensorInt::<1>::from([0, 2]).cast(burn_tensor::DType::U32);

    let one_hot_tensor: TestTensorInt<2> = tensor.one_hot(3);

    one_hot_tensor
        .into_data()
        .assert_eq(&TensorData::from([[1u32, 0, 0], [0, 0, 1]]), false);
}

// U64 is not supported on every backend
#[cfg(feature = "flex")]
#[test]
fn one_hot_fill_unsigned_index_above_i64_max_gives_off_value_row() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::from_data(
        TensorData::from([1u64, u64::MAX]),
        (&device, burn_tensor::DType::U64),
    );

    let one_hot_tensor: TestTensorInt<2> = tensor.one_hot_fill(3, 5.0, 1.0, -1);

    one_hot_tensor
        .into_data()
        .assert_eq(&TensorData::from([[1u64, 5, 1], [1, 1, 1]]), false);
}

// I8 is not supported on every backend
#[cfg(feature = "flex")]
#[test]
fn one_hot_fill_class_bounds_do_not_depend_on_index_dtype() {
    // 128 classes don't fit in i8: the bounds must still accept 127 and -128.
    let tensor = TestTensorInt::<1>::from([127, -128]).cast(burn_tensor::DType::I8);

    let one_hot_tensor: TestTensorInt<2> = tensor.one_hot_fill(128, 1.0, 0.0, -1);

    let mut expected = vec![0i8; 2 * 128];
    expected[127] = 1;
    expected[128] = 1;
    one_hot_tensor
        .into_data()
        .assert_eq(&TensorData::new(expected, [2, 128]), false);
}

// I8 is not supported on every backend
#[cfg(feature = "flex")]
#[test]
fn int_one_hot_accepts_class_count_above_index_dtype_max() {
    let tensor = TestTensorInt::<1>::from([127]).cast(burn_tensor::DType::I8);

    let one_hot_tensor: TestTensorInt<2> = tensor.one_hot(128);

    assert_eq!(one_hot_tensor.dims(), [1, 128]);
}
