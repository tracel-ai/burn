#[burn_tensor_testgen::testgen(sign)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_sign_ops_float() {
        let tensor = TestTensor::from([[-0.2, -1.0, 2.0], [3.0, 0.0, -5.0]]);

        let data_actual = tensor.sign().into_data();

        let data_expected = Data::from([[-1.0, -1.0, 1.0], [1.0, 0.0, -1.0]]);
        assert_eq!(data_actual, data_expected);
    }

    #[test]
    fn should_support_sign_ops_int() {
        let tensor = TestTensorInt::from([[-2, -1, 2], [3, 0, -5]]);

        let data_actual = tensor.sign().into_data();

        let data_expected = Data::from([[-1, -1, 1], [1, 0, -1]]);
        assert_eq!(data_actual, data_expected);
    }
}
