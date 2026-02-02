use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_sub_ops_int() {
    let data_1 = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let data_2 = TensorData::from([[6, 7, 8], [9, 10, 11]]);
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let output = tensor_1 - tensor_2;
    let expected = TensorData::from([[-6, -6, -6], [-6, -6, -6]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sub_broadcast_int() {
    let data_1 = TensorData::from([[0, 1, 2]]);
    let data_2 = TensorData::from([[3, 4, 5], [6, 7, 8]]);
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let output = tensor_1 - tensor_2;
    let expected = TensorData::from([[-3, -3, -3], [-6, -6, -6]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_sub_scalar_ops_int() {
    let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let scalar = 2;
    let tensor = TestTensorInt::<2>::from_data(data, &Default::default());

    let output = tensor - scalar;
    let expected = TensorData::from([[-2, -1, 0], [1, 2, 3]]);

    output.into_data().assert_eq(&expected, false);
}
