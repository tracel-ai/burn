#[burn_tensor_testgen::testgen(index_select)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_select_1d() {
        let tensor = TestTensor::from_data(Data::from([0.0, 1.0, 2.0]));
        let indexes = TestTensorInt::from_data(Data::from([1, 1, 0, 1, 2]));

        let output = tensor.index_select(indexes);

        assert_eq!(output.into_data(), Data::from([1.0, 1.0, 0.0, 1.0, 2.0]));
    }

    #[test]
    fn should_select_2d() {
        let tensor = TestTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]));
        let indexes = TestTensorInt::from_data(Data::from([[2, 1, 0, 0], [2, 0, 1, 2]]));

        let output = tensor.index_select(indexes);

        assert_eq!(
            output.into_data(),
            Data::from([[2.0, 1.0, 0.0, 0.0], [5.0, 3.0, 4.0, 5.0]])
        );
    }

    #[test]
    fn should_select_2d_only_1dim() {
        let tensor = TestTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]));
        let indexes = TestTensorInt::from_data(Data::from([[1, 2]])).reshape([2, 1]);

        let output = tensor.index_select(indexes);

        assert_eq!(output.into_data(), Data::from([[1.0], [5.0]]));
    }

    #[test]
    fn should_select_assign_1d() {
        let tensor = TestTensor::from_data(Data::from([0.0, 0.0, 0.0]));
        let values = TestTensor::from_data(Data::from([5.0, 4.0, 3.0]));
        let indexes = TestTensorInt::from_data(Data::from([1, 0, 2]));

        let output = tensor.index_select_assign(indexes, values);

        assert_eq!(output.into_data(), Data::from([4.0, 5.0, 3.0]));
    }

    #[test]
    fn should_select_assign_2d() {
        let tensor = TestTensor::from_data(Data::from([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]));
        let values = TestTensor::from_data(Data::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        let indexes = TestTensorInt::from_data(Data::from([[1, 0, 2], [1, 2, 0]]));

        let output = tensor.index_select_assign(indexes, values);

        assert_eq!(
            output.into_data(),
            Data::from([[2.0, 1.0, 3.0], [6.0, 4.0, 5.0]])
        );
    }
}
