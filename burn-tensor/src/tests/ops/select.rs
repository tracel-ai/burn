#[burn_tensor_testgen::testgen(select)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_select_1d() {
        let tensor = TestTensor::from_data([0.0, 1.0, 2.0]);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2]);

        let output = tensor.select(0, indices);

        assert_eq!(output.into_data(), Data::from([1.0, 1.0, 0.0, 1.0, 2.0]));
    }

    #[test]
    fn should_select_2d_dim0_same_num_dim() {
        let tensor = TestTensor::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indices = TestTensorInt::from_data(([1, 0]));

        let output = tensor.select(0, indices);

        assert_eq!(
            output.into_data(),
            Data::from([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]])
        );
    }

    #[test]
    fn should_select_2d_dim0_more_num_dim() {
        let tensor = TestTensor::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indices = TestTensorInt::from_data([1, 0, 1, 1]);

        let output = tensor.select(0, indices);

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
        let tensor = TestTensor::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2]);

        let output = tensor.select(1, indices);

        assert_eq!(
            output.into_data(),
            Data::from([[1.0, 1.0, 0.0, 1.0, 2.0], [4.0, 4.0, 3.0, 4.0, 5.0]])
        );
    }

    #[test]
    fn should_select_assign_1d() {
        let tensor = TestTensor::from_data([0.0, 1.0, 2.0]);
        let values = TestTensor::from_data([5.0, 4.0, 3.0, 2.0, 1.0]);
        let indices = TestTensorInt::from_data(Data::from([1, 1, 0, 1, 2]));

        let output = tensor.select_assign(0, indices, values);

        assert_eq!(output.into_data(), Data::from([3.0, 12.0, 3.0]));
    }

    #[test]
    fn should_select_assign_2d_dim0() {
        let tensor = TestTensor::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let values = TestTensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indices = TestTensorInt::from_data(Data::from([1, 0]));

        let output = tensor.select_assign(0, indices, values);

        assert_eq!(
            output.into_data(),
            Data::from([[4.0, 6.0, 8.0], [4.0, 6.0, 8.0]])
        );
    }

    #[test]
    fn should_select_assign_2d_dim1() {
        let tensor = TestTensor::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let values = TestTensor::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indices = TestTensorInt::from_data(Data::from([1, 0, 2]));

        let output = tensor.select_assign(1, indices, values);

        assert_eq!(
            output.into_data(),
            Data::from([[2.0, 2.0, 5.0], [8.0, 8.0, 11.0]])
        );
    }
}
