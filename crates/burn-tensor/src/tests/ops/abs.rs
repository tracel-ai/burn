#[burn_tensor_testgen::testgen(abs)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_abs_ops_float() {
        let tensor = TestTensor::from([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]);

        let data_actual = tensor.abs().into_data();

        let data_expected = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_abs_ops_int() {
        let tensor = TestTensorInt::from([[0, -1, 2], [3, 4, -5]]);

        let data_actual = tensor.abs().into_data();

        let data_expected = Data::from([[0, 1, 2], [3, 4, 5]]);
        assert_eq!(data_expected, data_actual);
    }
}
