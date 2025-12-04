use crate::*;
use burn_tensor::{DEFAULT_ATOL, DEFAULT_RTOL, TensorData};

#[test]
fn test_is_close() {
    let tensor1 = TestTensor::<2>::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]);
    let tensor2 = TestTensor::from([[0.0, 1.0, 0.0], [1.0, -1.0, 3.0]]) + 1e-9;

    let data_actual = tensor1
        .clone()
        .is_close(tensor2.clone(), None, None)
        .into_data();
    let defaults_expected = TensorData::from([[true, true, true], [true, true, false]]);
    defaults_expected.assert_eq(&data_actual, false);

    // Using the defaults.
    let data_actual = tensor1
        .is_close(tensor2, Some(DEFAULT_RTOL), Some(DEFAULT_ATOL))
        .into_data();
    defaults_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_all_close() {
    let tensor1 = TestTensor::<2>::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]);
    let tensor2 = TestTensor::from([[0.0, 1.0, 0.0], [1.0, -1.0, 3.0]]) + 1e-9;
    assert!(!tensor1.clone().all_close(tensor2.clone(), None, None));

    let tensor2 = TestTensor::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]) + 1e-9;
    assert!(tensor1.all_close(tensor2, None, None));

    // non finite values
    let inf_plus = TestTensor::<2>::from([[f32::INFINITY]]);
    let one = TestTensor::<2>::from([[1.]]);
    let inf_minus = TestTensor::<2>::from([[-f32::INFINITY]]);
    assert!(!inf_plus.clone().all_close(inf_minus.clone(), None, None));
    assert!(!one.clone().all_close(inf_minus.clone(), None, None));
    assert!(!one.all_close(inf_plus.clone(), None, None));
    assert!(inf_plus.clone().all_close(inf_plus, None, None));
}
