#[burn_tensor_testgen::testgen(gather_scatter)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_gather_1d_dim0() {
        let tensor = TestTensor::from_floats([0.0, 1.0, 2.0]);
        let indexes = TestTensorInt::from_ints([1, 1, 0, 1, 2]);

        let output = tensor.gather(0, indexes);

        assert_eq!(output.into_data(), Data::from([1.0, 1.0, 0.0, 1.0, 2.0]));
    }

    #[test]
    fn should_gather_2d_dim0() {
        let tensor = TestTensor::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indexes = TestTensorInt::from_ints([[0, 1, 0], [1, 0, 1]]);

        let output = tensor.gather(0, indexes);

        assert_eq!(
            output.into_data(),
            Data::from([[0.0, 4.0, 2.0], [3.0, 1.0, 5.0]])
        );
    }

    #[test]
    fn should_gather_2d_dim1() {
        let tensor = TestTensor::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indexes = TestTensorInt::from_ints([[2, 1, 0, 0], [2, 0, 1, 2]]);

        let output = tensor.gather(1, indexes);

        assert_eq!(
            output.into_data(),
            Data::from([[2.0, 1.0, 0.0, 0.0], [5.0, 3.0, 4.0, 5.0]])
        );
    }

    #[test]
    fn should_gather_2d_only_1dim() {
        let tensor = TestTensor::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indexes = TestTensorInt::from_ints([[1, 2]]).reshape([2, 1]);

        let output = tensor.gather(1, indexes);

        assert_eq!(output.into_data(), Data::from([[1.0], [5.0]]));
    }

    #[test]
    fn should_scatter_1d() {
        let tensor = TestTensor::from_floats([0.0, 0.0, 0.0]);
        let values = TestTensor::from_floats([5.0, 4.0, 3.0]);
        let indexes = TestTensorInt::from_ints([1, 0, 2]);

        let output = tensor.scatter(0, indexes, values);

        assert_eq!(output.into_data(), Data::from([4.0, 5.0, 3.0]));
    }

    #[test]
    fn should_scatter_2d_dim0() {
        let tensor = TestTensor::from_floats([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let values = TestTensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indexes = TestTensorInt::from_ints([[1, 0, 1], [1, 1, 0]]);

        let output = tensor.scatter(0, indexes, values);

        assert_eq!(
            output.into_data(),
            Data::from([[0.0, 2.0, 6.0], [5.0, 5.0, 3.0]])
        );
    }

    #[test]
    fn should_scatter_2d_dim1() {
        let tensor = TestTensor::from_floats([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let values = TestTensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indexes = TestTensorInt::from_ints([[1, 0, 2], [1, 2, 0]]);

        let output = tensor.scatter(1, indexes, values);

        assert_eq!(
            output.into_data(),
            Data::from([[2.0, 1.0, 3.0], [6.0, 4.0, 5.0]])
        );
    }
}
