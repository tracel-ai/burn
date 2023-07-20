#[burn_tensor_testgen::testgen(aggregation)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Shape, Tensor};

    #[test]
    fn test_should_mean() {
        let tensor = TestTensor::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let data_actual = tensor.mean().to_data();

        assert_eq!(data_actual, Data::from([15.0 / 6.0]));
    }

    #[test]
    fn test_should_sum() {
        let tensor = TestTensor::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let data_actual = tensor.sum().to_data();

        assert_eq!(data_actual, Data::from([15.0]));
    }

    #[test]
    fn test_should_mean_last_dim() {
        let tensor = TestTensor::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let data_actual = tensor.mean_dim(1).to_data();

        assert_eq!(data_actual, Data::from([[3.0 / 3.0], [12.0 / 3.0]]));
    }

    #[test]
    fn test_should_sum_last_dim() {
        let tensor = TestTensor::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let data_actual = tensor.sum_dim(1).to_data();

        assert_eq!(data_actual, Data::from([[3.0], [12.0]]));
    }

    #[test]
    fn test_should_sum_first_dim() {
        let tensor = TestTensor::from_data([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]);

        let data_actual = tensor.sum_dim(0).to_data();

        assert_eq!(data_actual, Data::from([[7.0, 3.0, 5.0]]));
    }

    #[test]
    fn test_should_mean_first_dim() {
        let tensor = TestTensor::from_data([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]);

        let data_actual = tensor.mean_dim(0).to_data();

        assert_eq!(data_actual, Data::from([[7.0 / 2.0, 3.0 / 2.0, 5.0 / 2.0]]));
    }

    #[test]
    fn test_should_sum_mid_dim_3d_non_contiguous_1() {
        let tensor = TestTensor::from_data([
            [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
            [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
        ]);

        let data_actual = tensor.swap_dims(0, 2).sum_dim(1).into_data();

        assert_eq!(
            data_actual,
            Data::new(vec![9.0, 7.0, -1.0, 3.0, 4.0, 5.0], Shape::new([3, 1, 2]))
        );
    }

    #[test]
    fn test_should_sum_mid_dim_3d_non_contiguous_2() {
        let tensor = TestTensor::from_data([
            [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
            [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
        ]);

        let data_actual = tensor.swap_dims(0, 1).sum_dim(1).into_data();

        assert_eq!(
            data_actual,
            Data::new(vec![5.0, 5.0, 3.0, 11.0, -3.0, 6.0], Shape::new([2, 1, 3]))
        );
    }
}
