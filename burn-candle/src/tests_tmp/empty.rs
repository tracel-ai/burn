// #[burn_tensor_testgen::testgen(empty)]
mod tests {
    use crate::CandleBackend;

    use super::*;
    use burn_tensor::{Bool, Int, Tensor};
    pub type TestBackend = CandleBackend<f32, i64>;

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
