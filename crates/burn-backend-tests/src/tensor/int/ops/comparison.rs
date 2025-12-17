use super::*;
use burn_tensor::TensorData;

#[test]
fn test_equal() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);
    let tensor_2 = TestTensorInt::<2>::from([[1, 1, 1], [4, 3, 5]]);

    let data_actual_cloned = tensor_1.clone().equal(tensor_2.clone());
    let data_actual_inplace = tensor_1.equal(tensor_2);

    let data_expected = TensorData::from([[false, true, false], [false, false, true]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_not_equal() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);
    let tensor_2 = TestTensorInt::<2>::from([[1, 1, 1], [4, 3, 5]]);

    let data_actual_cloned = tensor_1.clone().not_equal(tensor_2.clone());
    let data_actual_inplace = tensor_1.not_equal(tensor_2);

    let data_expected = TensorData::from([[true, false, true], [true, true, false]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_equal_elem() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 2, 5]]);

    let data_actual_cloned = tensor_1.clone().equal_elem(2);
    let data_actual_inplace = tensor_1.equal_elem(2);

    let data_expected = TensorData::from([[false, false, true], [false, true, false]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_not_equal_elem() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 2, 5]]);

    let data_actual_cloned = tensor_1.clone().not_equal_elem(2);
    let data_actual_inplace = tensor_1.not_equal_elem(2);

    let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn greater_elem() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    let data_actual_cloned = tensor_1.clone().greater_elem(4);
    let data_actual_inplace = tensor_1.greater_elem(4);

    let data_expected = TensorData::from([[false, false, false], [false, false, true]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_greater_equal_elem() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    let data_actual_cloned = tensor_1.clone().greater_equal_elem(4);
    let data_actual_inplace = tensor_1.greater_equal_elem(4);

    let data_expected = TensorData::from([[false, false, false], [false, true, true]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_greater() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);
    let tensor_2 = TestTensorInt::<2>::from([[1, 1, 1], [4, 3, 50]]);

    let data_actual_cloned = tensor_1.clone().greater(tensor_2.clone());
    let data_actual_inplace = tensor_1.greater(tensor_2);

    let data_expected = TensorData::from([[false, false, true], [false, true, false]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_greater_equal() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);
    let tensor_2 = TestTensorInt::<2>::from([[1, 1, 1], [4, 3, 50]]);

    let data_actual_cloned = tensor_1.clone().greater_equal(tensor_2.clone());
    let data_actual_inplace = tensor_1.greater_equal(tensor_2);

    let data_expected = TensorData::from([[false, true, true], [false, true, false]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_lower_elem() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    let data_actual_cloned = tensor_1.clone().lower_elem(4);
    let data_actual_inplace = tensor_1.lower_elem(4);

    let data_expected = TensorData::from([[true, true, true], [true, false, false]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_lower_equal_elem() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    let data_actual_cloned = tensor_1.clone().lower_equal_elem(4);
    let data_actual_inplace = tensor_1.lower_equal_elem(4);

    let data_expected = TensorData::from([[true, true, true], [true, true, false]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_lower() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);
    let tensor_2 = TestTensorInt::<2>::from([[1, 1, 1], [4, 3, 50]]);

    let data_actual_cloned = tensor_1.clone().lower(tensor_2.clone());
    let data_actual_inplace = tensor_1.lower(tensor_2);

    let data_expected = TensorData::from([[true, false, false], [true, false, true]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_lower_equal() {
    let tensor_1 = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);
    let tensor_2 = TestTensorInt::<2>::from([[1, 1, 1], [4, 3, 50]]);

    let data_actual_cloned = tensor_1.clone().lower_equal(tensor_2.clone());
    let data_actual_inplace = tensor_1.lower_equal(tensor_2);

    let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
    data_expected.assert_eq(&data_actual_cloned.into_data(), false);
    data_expected.assert_eq(&data_actual_inplace.into_data(), false);
}

#[test]
fn test_greater_broadcast() {
    // Test broadcasting with shape [1, 4] vs [4, 4]
    let device = Default::default();
    let data_1 = TensorData::from([[1, 2, 3, 4]]);
    let data_2 = TensorData::from([
        [0.5, 1.5, 2.5, 3.5],
        [1.5, 2.5, 3.5, 4.5],
        [2.5, 3.5, 4.5, 5.5],
        [3.5, 4.5, 5.5, 6.5],
    ]);
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let result = tensor_1.greater(tensor_2);

    let expected = TensorData::from([
        [true, true, true, true],
        [false, false, false, false],
        [false, false, false, false],
        [false, false, false, false],
    ]);
    expected.assert_eq(&result.into_data(), false);
}

#[test]
fn test_greater_equal_broadcast() {
    // Test broadcasting with shape [4, 1] vs [1, 4]
    let device = Default::default();
    let data_1 = TensorData::from([[1], [2], [3], [4]]);
    let data_2 = TensorData::from([[1, 2, 3, 4]]);
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let result = tensor_1.greater_equal(tensor_2);

    let expected = TensorData::from([
        [true, false, false, false],
        [true, true, false, false],
        [true, true, true, false],
        [true, true, true, true],
    ]);
    expected.assert_eq(&result.into_data(), false);
}

#[test]
fn test_equal_broadcast() {
    // Test broadcasting with different ranks
    let device = Default::default();
    let data_1 = TensorData::from([[2], [3], [4]]);
    let data_2 = TensorData::from([[2, 3, 4, 2]]);
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let result = tensor_1.equal(tensor_2);

    let expected = TensorData::from([
        [true, false, false, true],
        [false, true, false, false],
        [false, false, true, false],
    ]);
    expected.assert_eq(&result.into_data(), false);
}

#[test]
fn test_not_equal_broadcast() {
    // Test broadcasting with shape [3, 1] vs [1, 3]
    let device = Default::default();
    let data_1 = TensorData::from([[1], [2], [3]]);
    let data_2 = TensorData::from([[1, 2, 3]]);
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let result = tensor_1.not_equal(tensor_2);

    let expected = TensorData::from([
        [false, true, true],
        [true, false, true],
        [true, true, false],
    ]);
    expected.assert_eq(&result.into_data(), false);
}

#[test]
fn test_int_greater_broadcast() {
    let device = Default::default();
    let data_1 = TensorData::from([[1i32, 2, 3]]);
    let data_2 = TensorData::from([[0i32], [2], [4]]);
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let result = tensor_1.greater(tensor_2);

    let expected = TensorData::from([
        [true, true, true],
        [false, false, true],
        [false, false, false],
    ]);
    expected.assert_eq(&result.into_data(), false);
}

#[test]
fn test_int_lower_equal_broadcast() {
    let device = Default::default();
    let data_1 = TensorData::from([[2i32], [4]]);
    let data_2 = TensorData::from([[1i32, 2, 3]]);
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let result = tensor_1.lower_equal(tensor_2);

    let expected = TensorData::from([[false, true, true], [false, false, false]]);
    expected.assert_eq(&result.into_data(), false);
}
