use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_support_hypot_basic() {
    let data_a = TensorData::from([[3.0, 4.0], [5.0, 12.0]]);
    let data_b = TensorData::from([[4.0, 3.0], [12.0, 5.0]]);
    let tensor_a = TestTensor::<2>::from_data(data_a, &Default::default());
    let tensor_b = TestTensor::<2>::from_data(data_b, &Default::default());

    let result = tensor_a.hypot(tensor_b);
    let expected = TensorData::from([[5.0, 5.0], [13.0, 13.0]]);

    result
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_hypot_broadcast() {
    let data_a = TensorData::from([[3.0, 4.0, 5.0]]);
    let data_b = TensorData::from([[4.0], [3.0]]);
    let tensor_a = TestTensor::<2>::from_data(data_a, &Default::default());
    let tensor_b = TestTensor::<2>::from_data(data_b, &Default::default());

    let result = tensor_a.hypot(tensor_b);
    let expected = TensorData::from([[5.0, 5.656854, 6.404], [4.2426405, 5.0, 5.831]]);

    result
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_hypot_zero() {
    let data_a = TensorData::from([[0.0, 3.0], [0.0, 0.0]]);
    let data_b = TensorData::from([[4.0, 0.0], [0.0, 5.0]]);
    let tensor_a = TestTensor::<2>::from_data(data_a, &Default::default());
    let tensor_b = TestTensor::<2>::from_data(data_b, &Default::default());

    let result = tensor_a.hypot(tensor_b);
    let expected = TensorData::from([[4.0, 3.0], [0.0, 5.0]]);

    result
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
