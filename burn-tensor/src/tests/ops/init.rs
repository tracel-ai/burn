#[burn_tensor_testgen::testgen(init)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Data, Int, Tensor};

    #[test]
    fn should_support_float_empty() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2>::empty(shape);
        assert_eq!(tensor.shape(), shape.into())
    }

    #[test]
    fn should_support_int_empty() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2, Int>::empty(shape);
        assert_eq!(tensor.shape(), shape.into())
    }

    #[test]
    fn should_support_float_zeros() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2>::zeros(shape);
        assert_eq!(tensor.shape(), shape.into());
        assert_eq!(tensor.to_data(), Data::from([[0., 0.], [0., 0.]]))
    }

    #[test]
    fn should_support_int_zeros() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2, Int>::zeros(shape);
        assert_eq!(tensor.shape(), shape.into());
        assert_eq!(tensor.to_data(), Data::from([[0, 0], [0, 0]]))
    }

    #[test]
    fn should_support_float_ones() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2>::ones(shape);
        assert_eq!(tensor.shape(), shape.into());
        assert_eq!(tensor.to_data(), Data::from([[1., 1.], [1., 1.]]))
    }

    #[test]
    fn should_support_int_ones() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2, Int>::ones(shape);
        assert_eq!(tensor.shape(), shape.into());
        assert_eq!(tensor.to_data(), Data::from([[1, 1], [1, 1]]))
    }

    #[test]
    fn should_support_bool_empty() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2, Bool>::empty(shape);
        assert_eq!(tensor.shape(), shape.into())
    }
}
