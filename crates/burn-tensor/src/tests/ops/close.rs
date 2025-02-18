#[burn_tensor_testgen::testgen(close)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData, DEFAULT_ATOL, DEFAULT_RTOL};

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
    }
}
