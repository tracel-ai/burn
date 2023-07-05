#[burn_tensor_testgen::testgen(maxmin)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_max_dim_2d() {
        let tensor = TestTensor::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output_actual = tensor.max_dim(1);

        let output_expected = Data::from([[2.], [5.]]);
        assert_eq!(output_expected, output_actual.into_data());
    }

    #[test]
    fn test_max_dim_with_indices_2d() {
        let tensor = TestTensor::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let (output_actual, index_actual) = tensor.max_dim_with_indices(1);

        let output_expected = Data::from([[2.], [5.]]);
        let index_expected = Data::from([[2], [2]]);

        assert_eq!(output_expected, output_actual.into_data());
        assert_eq!(index_expected, index_actual.into_data());
    }

    #[test]
    fn test_min_dim_2d() {
        let tensor = TestTensor::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output_actual = tensor.min_dim(1);

        let output_expected = Data::from([[0.], [3.]]);
        assert_eq!(output_expected, output_actual.into_data());
    }

    #[test]
    fn test_min_dim_with_indices_2d() {
        let tensor = TestTensor::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let (output_actual, index_actual) = tensor.min_dim_with_indices(1);

        let output_expected = Data::from([[0.], [3.]]);
        let index_expected = Data::from([[0], [0]]);

        assert_eq!(output_expected, output_actual.into_data());
        assert_eq!(index_expected, index_actual.into_data());
    }
}
