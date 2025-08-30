#[cfg(test)]
mod tests {
    use super::super::view::TensorView;
    use crate::TestBackend;
    use burn_tensor::Tensor;

    #[test]
    fn test_tensor_view_float() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        let view = TensorView::from_float(&tensor);
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_view_int() {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 2, burn_tensor::Int>::from_data([[1, 2], [3, 4]], &device);

        let view = TensorView::from_int(&tensor);
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_view_bool() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2, burn_tensor::Bool>::from_data(
            [[true, false], [false, true]],
            &device,
        );

        let view = TensorView::from_bool(&tensor);
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }
}
