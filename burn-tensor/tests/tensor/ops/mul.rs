use super::super::TestBackend;
use burn_tensor::{Data, Tensor};

#[test]
fn should_support_mul_ops() {
    let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data_2 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_1 = Tensor::<2, TestBackend>::from_data(data_1);
    let tensor_2 = Tensor::<2, TestBackend>::from_data(data_2);

    let output = tensor_1 * tensor_2;

    let data_actual = output.into_data();
    let data_expected = Data::from([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);
    assert_eq!(data_expected, data_actual);
}

#[test]
fn should_support_mul_scalar_ops() {
    let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let scalar = 2.0;
    let tensor = Tensor::<2, TestBackend>::from_data(data);

    let output = tensor * scalar;

    let data_actual = output.into_data();
    let data_expected = Data::from([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]);
    assert_eq!(data_expected, data_actual);
}
