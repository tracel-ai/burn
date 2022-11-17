#[burn_tensor_testgen::testgen(aggregation)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_should_mean() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.mean().to_data();

        assert_eq!(data_actual, Data::from([15.0 / 6.0]));
    }

    #[test]
    fn test_should_sum() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.sum().to_data();

        assert_eq!(data_actual, Data::from([15.0]));
    }

    #[test]
    fn test_should_mean_dim() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.mean_dim(1).to_data();

        assert_eq!(data_actual, Data::from([[3.0 / 3.0], [12.0 / 3.0]]));
    }

    #[test]
    fn test_should_sum_dim() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.sum_dim(1).to_data();

        assert_eq!(data_actual, Data::from([[3.0], [12.0]]));
    }
}
