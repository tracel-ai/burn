use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_sub_ops() {
    let data_1 = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let data_2 = TensorData::from([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensor::<2>::from_data(data_2, &device);

    let output = tensor_1 - tensor_2;
    let expected = TensorData::from([[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sub_broadcast() {
    let data_1 = TensorData::from([[0.0, 1.0, 2.0]]);
    let data_2 = TensorData::from([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);
    let device = Default::default();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensor::<2>::from_data(data_2, &device);

    let output = tensor_1 - tensor_2;
    let expected = TensorData::from([[-3.0, -3.0, -3.0], [-6.0, -6.0, -6.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_sub_scalar_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let scalar = 2.0;
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor - scalar;
    let expected = TensorData::from([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);

    output.into_data().assert_eq(&expected, false);
}
