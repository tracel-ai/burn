#[burn_tensor_testgen::testgen(maxmin)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_max_dim_2d() {
        let tensor = TestTensor::from_floats_devauto([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output_actual = tensor.max_dim(1);

        let output_expected = Data::from([[2.], [5.]]);
        assert_eq!(output_expected, output_actual.into_data());
    }

    #[test]
    fn test_max_dim_with_indices_2d_with_dim_0th() {
        let tensor = TestTensor::from_floats_devauto([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let (output_actual, index_actual) = tensor.max_dim_with_indices(0);

        let output_expected = Data::from([[3., 4., 5.]]);
        let index_expected = Data::from([[1, 1, 1]]);

        assert_eq!(output_expected, output_actual.into_data());
        assert_eq!(index_expected, index_actual.into_data());
    }

    #[test]
    fn test_max_dim_with_indices_2d() {
        let tensor = TestTensor::from_floats_devauto([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let (output_actual, index_actual) = tensor.max_dim_with_indices(1);

        let output_expected = Data::from([[2.], [5.]]);
        let index_expected = Data::from([[2], [2]]);

        assert_eq!(output_expected, output_actual.into_data());
        assert_eq!(index_expected, index_actual.into_data());
    }

    #[test]
    fn test_min_dim_2d() {
        let tensor = TestTensor::from_floats_devauto([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output_actual = tensor.min_dim(1);

        let output_expected = Data::from([[0.], [3.]]);
        assert_eq!(output_expected, output_actual.into_data());
    }

    #[test]
    fn test_min_dim_with_indices_2d() {
        let tensor = TestTensor::from_floats_devauto([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let (output_actual, index_actual) = tensor.min_dim_with_indices(1);

        let output_expected = Data::from([[0.], [3.]]);
        let index_expected = Data::from([[0], [0]]);

        assert_eq!(output_expected, output_actual.into_data());
        assert_eq!(index_expected, index_actual.into_data());
    }

    #[test]
    fn test_sum_dim_2d() {
        let tensor = TestTensor::from_floats_devauto([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output_actual = tensor.clone().sum_dim(1);

        let output_expected = Data::from([[3.], [12.]]);
        assert_eq!(output_expected, output_actual.into_data());

        let output_actual = tensor.sum_dim(0);

        let output_expected = Data::from([[3., 5., 7.]]);
        assert_eq!(output_expected, output_actual.into_data());
    }

    #[test]
    fn test_mean_dim_2d() {
        let tensor = TestTensor::from_floats_devauto([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output_actual = tensor.clone().mean_dim(1);

        let output_expected = Data::from([[1.], [4.]]);
        assert_eq!(output_expected, output_actual.into_data());

        let output_actual = tensor.mean_dim(0);

        let output_expected = Data::from([[1.5, 2.5, 3.5]]);
        assert_eq!(output_expected, output_actual.into_data());
    }

    #[test]
    fn test_min_dim_2d_with_0th_dim() {
        let tensor = TestTensor::from_floats_devauto([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let output_actual = tensor.min_dim(0);

        let output_expected = Data::from([[0., 1., 2.]]);
        assert_eq!(output_expected, output_actual.into_data());
    }

    #[test]
    fn test_max_dim_2d_with_0th_dim() {
        let tensor = TestTensor::from_floats_devauto([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output_actual = tensor.max_dim(0);

        let output_expected = Data::from([[3., 4., 5.]]);
        assert_eq!(output_expected, output_actual.into_data());
    }

    #[test]
    fn test_min_dim_with_indices_2d_with_0th_dim() {
        let tensor = TestTensor::from_floats_devauto([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let (output_actual, index_actual) = tensor.min_dim_with_indices(0);

        let output_expected = Data::from([[0., 1., 2.]]);
        let index_expected = Data::from([[0, 0, 0]]);

        assert_eq!(output_expected, output_actual.into_data());
        assert_eq!(index_expected, index_actual.into_data());
    }
}
