#[burn_tensor_testgen::testgen(arg)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_argmax_2d_dim0() {
        let data = Data::from([[10.0, 11.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.argmax(0);

        let data_expected = Data::from([[0, 0, 1]]);
        assert_eq!(data_expected, data_actual.to_data());
    }

    #[test]
    fn test_argmin_2d_dim0() {
        let data = Data::from([[10.0, 11.0, 2.0], [30.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.argmin(0);

        let data_expected = Data::from([[0, 1, 0]]);
        assert_eq!(data_expected, data_actual.to_data());
    }

    #[test]
    fn test_argmax_2d_dim1() {
        let data = Data::from([[10.0, 11.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.argmax(1);

        let data_expected = Data::from([[1], [2]]);
        assert_eq!(data_expected, data_actual.to_data());
    }

    #[test]
    fn test_argmin_2d_dim1() {
        let data = Data::from([[10.0, 11.0, 2.0], [30.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.argmin(1);

        let data_expected = Data::from([[2], [1]]);
        assert_eq!(data_expected, data_actual.to_data());
    }
}
