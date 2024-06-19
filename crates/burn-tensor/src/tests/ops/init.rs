#[burn_tensor_testgen::testgen(init)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Bool, Int, Tensor, TensorData};

    #[test]
    fn should_support_float_empty() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2>::empty(shape, &Default::default());
        assert_eq!(tensor.shape(), shape.into())
    }

    #[test]
    fn should_support_int_empty() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2, Int>::empty(shape, &Default::default());
        assert_eq!(tensor.shape(), shape.into())
    }

    #[test]
    fn should_support_float_zeros() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2>::zeros(shape, &Default::default());
        assert_eq!(tensor.shape(), shape.into());
        let expected =
            TensorData::from([[0., 0.], [0., 0.]]).convert::<<TestBackend as Backend>::FloatElem>();

        tensor.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn should_support_int_zeros() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2, Int>::zeros(shape, &Default::default());
        assert_eq!(tensor.shape(), shape.into());
        let expected =
            TensorData::from([[0, 0], [0, 0]]).convert::<<TestBackend as Backend>::IntElem>();

        tensor.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn should_support_float_ones() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2>::ones(shape, &Default::default());
        assert_eq!(tensor.shape(), shape.into());
        let expected =
            TensorData::from([[1., 1.], [1., 1.]]).convert::<<TestBackend as Backend>::FloatElem>();

        tensor.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn should_support_int_ones() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2, Int>::ones(shape, &Default::default());
        assert_eq!(tensor.shape(), shape.into());
        let expected =
            TensorData::from([[1, 1], [1, 1]]).convert::<<TestBackend as Backend>::IntElem>();

        tensor.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn should_support_bool_empty() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2, Bool>::empty(shape, &Default::default());
        assert_eq!(tensor.shape(), shape.into())
    }
}
