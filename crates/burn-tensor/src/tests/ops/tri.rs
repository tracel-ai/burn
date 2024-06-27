#[burn_tensor_testgen::testgen(tri)]
mod tests {
    use super::*;
    use burn_tensor::{Int, Shape, Tensor, TensorData};

    #[test]
    fn test_triu() {
        let tensor: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]),
            &Default::default(),
        );
        let output = tensor.triu(0);
        let expected = TensorData::from([[1., 1., 1.], [0., 1., 1.], [0., 0., 1.]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_triu_positive_diagonal() {
        let tensor: Tensor<TestBackend, 2, Int> = Tensor::from_data(
            TensorData::from([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            &Default::default(),
        );

        let output = tensor.triu(1);
        let expected = TensorData::from([[0, 1, 1], [0, 0, 1], [0, 0, 0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_triu_negative_diagonal() {
        let tensor: Tensor<TestBackend, 2, Int> = Tensor::from_data(
            TensorData::from([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            &Default::default(),
        );

        let output = tensor.triu(-1);
        let expected = TensorData::from([[1, 1, 1], [1, 1, 1], [0, 1, 1]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_triu_batch_tensors() {
        let tensor: Tensor<TestBackend, 4, Int> = Tensor::from_data(
            TensorData::from([
                [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
                [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
            ]),
            &Default::default(),
        );
        let output = tensor.triu(1);
        let expected = TensorData::from([
            [[[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]],
            [[[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    #[should_panic]
    fn test_triu_too_few_dims() {
        let tensor: Tensor<TestBackend, 1, Int> =
            Tensor::from_data(TensorData::from([1, 2, 3]), &Default::default());
        let output = tensor.triu(0);
    }

    #[test]
    fn test_tril() {
        let tensor: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::from([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]),
            &Default::default(),
        );
        let output = tensor.tril(0);
        let expected = TensorData::from([[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_tril_positive_diagonal() {
        let tensor: Tensor<TestBackend, 2, Int> = Tensor::from_data(
            TensorData::from([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            &Default::default(),
        );

        let output = tensor.tril(1);
        let expected = TensorData::from([[1, 1, 0], [1, 1, 1], [1, 1, 1]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_tril_negative_diagonal() {
        let tensor: Tensor<TestBackend, 2, Int> = Tensor::from_data(
            TensorData::from([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            &Default::default(),
        );

        let output = tensor.tril(-1);
        let expected = TensorData::from([[0, 0, 0], [1, 0, 0], [1, 1, 0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_tril_batch_tensors() {
        let tensor: Tensor<TestBackend, 4, Int> = Tensor::from_data(
            TensorData::from([
                [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
                [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
            ]),
            &Default::default(),
        );
        let output = tensor.tril(1);
        let expected = TensorData::from([
            [[[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]],
            [[[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    #[should_panic]
    fn test_tril_too_few_dims() {
        let tensor: Tensor<TestBackend, 1, Int> =
            Tensor::from_data(TensorData::from([1, 2, 3]), &Default::default());
        let output = tensor.tril(0);
    }
}
