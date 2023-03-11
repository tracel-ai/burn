#[burn_tensor_testgen::testgen(select)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_select_1d() {
        let tensor = TestTensor::from_data(Data::from([0.0, 1.0, 2.0]));
        let indexes = TestTensorInt::from_data(Data::from([1, 1, 0, 1, 2]));

        let output = tensor.select(0, indexes);

        assert_eq!(output.into_data(), Data::from([1.0, 1.0, 0.0, 1.0, 2.0]));
    }

    #[test]
    fn should_select_2d_dim0() {
        let tensor = TestTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]));
        let indexes = TestTensorInt::from_data(Data::from([1, 0]));

        let output_dim0 = tensor.select(0, indexes);

        assert_eq!(
            output_dim0.into_data(),
            Data::from([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]])
        );
    }

    #[test]
    fn should_select_2d_dim1() {
        let tensor = TestTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]));
        let indexes = TestTensorInt::from_data(Data::from([1, 1, 0, 1, 2]));

        let output_dim0 = tensor.select(1, indexes);

        assert_eq!(
            output_dim0.into_data(),
            Data::from([[1.0, 1.0, 0.0, 1.0, 2.0], [4.0, 4.0, 3.0, 4.0, 5.0]])
        );
    }

    #[test]
    fn should_select_assign_1d() {
        let tensor = TestTensor::from_data(Data::from([0.0, 1.0, 2.0]));
        let values = TestTensor::from_data(Data::from([5.0, 4.0, 3.0, 2.0, 1.0]));
        let indexes = TestTensorInt::from_data(Data::from([1, 1, 0, 1, 2]));

        let output = tensor.select_assign(0, indexes, values);

        assert_eq!(output.into_data(), Data::from([3.0, 12.0, 3.0]));
    }
}
