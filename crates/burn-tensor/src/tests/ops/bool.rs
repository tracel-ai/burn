#[burn_tensor_testgen::testgen(bool)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_from_float() {
        let tensor1 = TestTensor::from([[0.0, 43.0, 0.0], [2.0, -4.2, 31.33]]);
        let data_actual = tensor1.bool().into_data();
        let data_expected = Data::from([[false, true, false], [true, true, true]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_from_int() {
        let tensor1 = TestTensorInt::from([[0, 43, 0], [2, -4, 31]]);
        let data_actual = tensor1.bool().into_data();
        let data_expected = Data::from([[false, true, false], [true, true, true]]);
        assert_eq!(data_expected, data_actual);
    }
}
