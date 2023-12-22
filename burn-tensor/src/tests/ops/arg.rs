#[burn_tensor_testgen::testgen(arg)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_argmax_2d_dim0() {
        let tensor = TestTensor::from([[10.0, 11.0, 2.0], [3.0, 4.0, 5.0]]);

        let data_actual = tensor.argmax(0);

        let data_expected = Data::from([[0, 0, 1]]);
        assert_eq!(data_expected, data_actual.into_data());
    }

    #[test]
    fn test_argmin_2d_dim0() {
        let tensor = TestTensor::from([[10.0, 11.0, 2.0], [30.0, 4.0, 5.0]]);

        let data_actual = tensor.argmin(0);

        let data_expected = Data::from([[0, 1, 0]]);
        assert_eq!(data_expected, data_actual.into_data());
    }

    #[test]
    fn test_argmax_2d_dim0_int() {
        let tensor = TestTensorInt::from([[10, 11, 2], [3, 4, 5]]);

        let data_actual = tensor.argmax(0);

        let data_expected = Data::from([[0, 0, 1]]);
        assert_eq!(data_expected, data_actual.into_data());
    }

    #[test]
    fn test_argmin_2d_dim0_int() {
        let tensor = TestTensorInt::from([[10, 11, 2], [30, 4, 5]]);

        let data_actual = tensor.argmin(0);

        let data_expected = Data::from([[0, 1, 0]]);
        assert_eq!(data_expected, data_actual.into_data());
    }

    #[test]
    fn test_argmax_2d_dim1() {
        let tensor = TestTensor::from([[10.0, 11.0, 2.0], [3.0, 4.0, 5.0]]);

        let data_actual = tensor.argmax(1);

        let data_expected = Data::from([[1], [2]]);
        assert_eq!(data_expected, data_actual.into_data());
    }

    #[test]
    fn test_argmin_2d_dim1() {
        let tensor = TestTensor::from([[10.0, 11.0, 2.0], [30.0, 4.0, 5.0]]);

        let data_actual = tensor.argmin(1);

        let data_expected = Data::from([[2], [1]]);
        assert_eq!(data_expected, data_actual.into_data());
    }
}
