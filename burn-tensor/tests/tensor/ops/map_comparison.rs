use super::super::TestBackend;
use burn_tensor::{Data, Tensor};

#[test]
fn test_greater_scalar() {
    let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_1 = Tensor::<2, TestBackend>::from_data(data_1);

    let data_actual = tensor_1.greater_scalar(&4.0);

    let data_expected = Data::from([[false, false, false], [false, false, true]]);
    assert_eq!(data_expected, data_actual.to_data());
}

#[test]
fn test_greater_equal_scalar() {
    let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_1 = Tensor::<2, TestBackend>::from_data(data_1);

    let data_actual = tensor_1.greater_equal_scalar(&4.0);

    let data_expected = Data::from([[false, false, false], [false, true, true]]);
    assert_eq!(data_expected, data_actual.to_data());
}

#[test]
fn test_greater() {
    let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data_2 = Data::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]);
    let tensor_1 = Tensor::<2, TestBackend>::from_data(data_1);
    let tensor_2 = Tensor::<2, TestBackend>::from_data(data_2);

    let data_actual = tensor_1.greater(&tensor_2);

    let data_expected = Data::from([[false, false, true], [false, true, false]]);
    assert_eq!(data_expected, data_actual.to_data());
}

#[test]
fn test_greater_equal() {
    let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data_2 = Data::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]);
    let tensor_1 = Tensor::<2, TestBackend>::from_data(data_1);
    let tensor_2 = Tensor::<2, TestBackend>::from_data(data_2);

    let data_actual = tensor_1.greater_equal(&tensor_2);

    let data_expected = Data::from([[false, true, true], [false, true, false]]);
    assert_eq!(data_expected, data_actual.to_data());
}

#[test]
fn test_lower_scalar() {
    let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_1 = Tensor::<2, TestBackend>::from_data(data_1);

    let data_actual = tensor_1.lower_scalar(&4.0);

    let data_expected = Data::from([[true, true, true], [true, false, false]]);
    assert_eq!(data_expected, data_actual.to_data());
}

#[test]
fn test_lower_equal_scalar() {
    let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_1 = Tensor::<2, TestBackend>::from_data(data_1);

    let data_actual = tensor_1.lower_equal_scalar(&4.0);

    let data_expected = Data::from([[true, true, true], [true, true, false]]);
    assert_eq!(data_expected, data_actual.to_data());
}

#[test]
fn test_lower() {
    let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data_2 = Data::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]);
    let tensor_1 = Tensor::<2, TestBackend>::from_data(data_1);
    let tensor_2 = Tensor::<2, TestBackend>::from_data(data_2);

    let data_actual = tensor_1.lower(&tensor_2);

    let data_expected = Data::from([[true, false, false], [true, false, true]]);
    assert_eq!(data_expected, data_actual.to_data());
}

#[test]
fn test_lower_equal() {
    let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data_2 = Data::from([[1.0, 1.0, 1.0], [4.0, 3.0, 50.0]]);
    let tensor_1 = Tensor::<2, TestBackend>::from_data(data_1);
    let tensor_2 = Tensor::<2, TestBackend>::from_data(data_2);

    let data_actual = tensor_1.lower_equal(&tensor_2);

    let data_expected = Data::from([[true, true, false], [true, false, true]]);
    assert_eq!(data_expected, data_actual.to_data());
}
