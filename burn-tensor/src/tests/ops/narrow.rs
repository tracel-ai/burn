#[burn_tensor_testgen::testgen(narrow)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Shape, Tensor};

    #[test]
    fn test_narrow() {
        let tensor: Tensor<TestBackend, 2> =
            Tensor::from_data(Data::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]));

        let output = tensor.clone().narrow(0, 0, 2);
        assert_eq!(output.shape(), Shape::from([2, 3]));
        output
            .to_data()
            .assert_approx_eq(&Data::from([[1., 2., 3.], [4., 5., 6.]]), 3);

        let output = tensor.clone().narrow(1, 1, 2);
        assert_eq!(output.shape(), Shape::from([3, 2]));
        output
            .to_data()
            .assert_approx_eq(&Data::from([[2., 3.], [5., 6.], [8., 9.]]), 3);
    }

    #[test]
    #[should_panic]
    fn test_narrow_invalid_dim() {
        let tensor: Tensor<TestBackend, 2> =
            Tensor::from_data(Data::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]));

        let output = tensor.narrow(2, 0, 2);
    }

    #[test]
    #[should_panic]
    fn test_narrow_invalid_start() {
        let tensor: Tensor<TestBackend, 2> =
            Tensor::from_data(Data::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]));

        let output = tensor.narrow(0, 3, 2);
    }

    #[test]
    #[should_panic]
    fn test_narrow_invalid_zero_length() {
        let tensor: Tensor<TestBackend, 2> =
            Tensor::from_data(Data::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]));

        let output = tensor.narrow(0, 1, 0);
    }

    #[test]
    #[should_panic]
    fn test_narrow_invalid_length() {
        let tensor: Tensor<TestBackend, 2> =
            Tensor::from_data(Data::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]));

        let output = tensor.narrow(0, 0, 4);
    }
}
