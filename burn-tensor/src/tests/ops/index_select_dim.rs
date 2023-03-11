#[burn_tensor_testgen::testgen(index_select_dim)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_select_1d() {
        let tensor = TestTensor::from_data(Data::from([0.0, 1.0, 2.0]));
        let indexes = TestTensorInt::from_data(Data::from([1, 1, 0, 1, 2]));

        let output = tensor.index_select_dim(0, indexes);

        assert_eq!(output.into_data(), Data::from([1.0, 1.0, 0.0, 1.0, 2.0]));
    }

    #[test]
    fn should_select_2d_dim0_same_num_dim() {
        let tensor = TestTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]));
        let indexes = TestTensorInt::from_data(Data::from([1, 0]));

        let output = tensor.index_select_dim(0, indexes);

        assert_eq!(
            output.into_data(),
            Data::from([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]])
        );
    }

    #[test]
    fn should_select_2d_dim0_more_num_dim() {
        let tensor = TestTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]));
        let indexes = TestTensorInt::from_data(Data::from([1, 0, 1, 1]));

        let output = tensor.index_select_dim(0, indexes);

        assert_eq!(
            output.into_data(),
            Data::from([
                [3.0, 4.0, 5.0],
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0]
            ])
        );
    }

    #[test]
    fn should_select_2d_dim1() {
        let tensor = TestTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]));
        let indexes = TestTensorInt::from_data(Data::from([1, 1, 0, 1, 2]));

        let output = tensor.index_select_dim(1, indexes);

        assert_eq!(
            output.into_data(),
            Data::from([[1.0, 1.0, 0.0, 1.0, 2.0], [4.0, 4.0, 3.0, 4.0, 5.0]])
        );
    }

    #[test]
    fn should_select_assign_1d() {
        let tensor = TestTensor::from_data(Data::from([0.0, 1.0, 2.0]));
        let values = TestTensor::from_data(Data::from([5.0, 4.0, 3.0, 2.0, 1.0]));
        let indexes = TestTensorInt::from_data(Data::from([1, 1, 0, 1, 2]));

        let output = tensor.index_select_dim_assign(0, indexes, values);

        assert_eq!(output.into_data(), Data::from([3.0, 12.0, 3.0]));
    }

    #[test]
    fn should_select_assign_2d_dim0() {
        let tensor = TestTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]));
        let values = TestTensor::from_data(Data::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        let indexes = TestTensorInt::from_data(Data::from([1, 0]));

        let output = tensor.index_select_dim_assign(0, indexes, values);

        assert_eq!(
            output.into_data(),
            Data::from([[4.0, 6.0, 8.0], [4.0, 6.0, 8.0]])
        );
    }
}
