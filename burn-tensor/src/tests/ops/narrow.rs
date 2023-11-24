#[burn_tensor_testgen::testgen(narrow)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Shape, Tensor};

    #[test]
    fn test_narrow() {
        let tensor: Tensor<TestBackend, 2> =
            Tensor::from_data(Data::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]));

        let output = tensor.narrow(0, 0, 2);
        assert_eq!(output.shape(), Shape::from([2, 3]));
    }
}
