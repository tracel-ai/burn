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
