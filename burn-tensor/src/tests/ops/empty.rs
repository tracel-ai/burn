#[burn_tensor_testgen::testgen(empty)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Int, Tensor};

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
    fn should_support_bool_empty() {
        let shape = [2, 2];
        let tensor = Tensor::<TestBackend, 2, Bool>::empty(shape);
        assert_eq!(tensor.shape(), shape.into())
    }
}
