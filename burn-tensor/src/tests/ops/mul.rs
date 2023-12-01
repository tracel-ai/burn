#[burn_tensor_testgen::testgen(mul)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn should_support_mul_ops() {
        let data_1 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let data_2 = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2>::from_data(data_2);

        let output = tensor_1 * tensor_2;

        let data_actual = output.into_data();
        let data_expected = Data::from([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_mul_broadcast() {
        let data_1 = Data::from([[0.0, 1.0, 2.0]]);
        let data_2 = Data::from([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);
        let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2>::from_data(data_2);

        let data_actual = (tensor_1 * tensor_2).into_data();

        let data_expected = Data::from([[0.0, 4.0, 10.0], [0.0, 7.0, 16.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_mul_broadcast_2_dims() {
        let tensor_1: Tensor<TestBackend, 2> = Tensor::from_data([0.0, 1.0, 2.0]).reshape([3, 1]);
        let tensor_2: Tensor<TestBackend, 2> = Tensor::from_data([3.0, 4.0, 5.0]).reshape([1, 3]);

        let data_actual = (tensor_1 * tensor_2).into_data();

        let data_expected = Data::from([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_mul_scalar_ops() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let scalar = 2.0;
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let output = tensor * scalar;

        let data_actual = output.into_data();
        let data_expected = Data::from([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_mul_ops_int() {
        let data_1 = Data::from([[0, 1, 2], [3, 4, 5]]);
        let data_2 = Data::from([[0, 1, 2], [3, 4, 5]]);
        let tensor_1 = Tensor::<TestBackend, 2, Int>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2, Int>::from_data(data_2);

        let output = tensor_1 * tensor_2;

        let data_actual = output.into_data();
        let data_expected = Data::from([[0, 1, 4], [9, 16, 25]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_mul_broadcast_int() {
        let data_1 = Data::from([[0, 1, 2]]);
        let data_2 = Data::from([[3, 4, 5], [6, 7, 8]]);
        let tensor_1 = Tensor::<TestBackend, 2, Int>::from_data(data_1);
        let tensor_2 = Tensor::<TestBackend, 2, Int>::from_data(data_2);

        let data_actual = (tensor_1 * tensor_2).into_data();

        let data_expected = Data::from([[0, 4, 10], [0, 7, 16]]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_mul_scalar_ops_int() {
        let data = Data::from([[0, 1, 2], [3, 4, 5]]);
        let scalar = 2;
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data);

        let output = tensor * scalar;

        let data_actual = output.into_data();
        let data_expected = Data::from([[0, 2, 4], [6, 8, 10]]);
        assert_eq!(data_expected, data_actual);
    }
}
