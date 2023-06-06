#[burn_tensor_testgen::testgen(squeeze)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Shape, Tensor};

    /// Test if the function can successfully squeeze the size 1 dimension of a 3D tensor.
    #[test]
    fn should_squeeze() {
        let tensor = Tensor::<TestBackend, 3>::ones(Shape::new([2, 1, 4]));
        let squeezed_tensor: Tensor<TestBackend, 2> = tensor.squeeze(1);
        let expected_shape = Shape::new([2, 4]);
        assert_eq!(squeezed_tensor.shape(), expected_shape);
    }
    /// Test if the function can successfully squeeze the first size 1 dimension of a 4D tensor.
    #[test]
    fn should_squeeze_first() {
        let tensor = Tensor::<TestBackend, 4>::ones(Shape::new([1, 3, 4, 5]));
        let squeezed_tensor: Tensor<TestBackend, 3> = tensor.squeeze(0);
        let expected_shape = Shape::new([3, 4, 5]);
        assert_eq!(squeezed_tensor.shape(), expected_shape);
    }
    /// Test if the function can successfully squeeze the last size 1 dimension of a 4D tensor.
    #[test]
    fn should_squeeze_last() {
        let tensor = Tensor::<TestBackend, 4>::ones(Shape::new([2, 3, 4, 1]));
        let squeezed_tensor: Tensor<TestBackend, 3> = tensor.squeeze(3);
        let expected_shape = Shape::new([2, 3, 4]);
        assert_eq!(squeezed_tensor.shape(), expected_shape);
    }
    /// Test if the function panics when the squeezed dimension is not of size 1.
    #[test]
    #[should_panic]
    fn should_squeeze_panic() {
        let tensor = Tensor::<TestBackend, 4>::ones(Shape::new([2, 3, 4, 5]));
        let squeezed_tensor: Tensor<TestBackend, 3> = tensor.squeeze(2);
    }
}
