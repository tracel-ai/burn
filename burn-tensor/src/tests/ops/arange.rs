#[burn_tensor_testgen::testgen(arange)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn test_arange_devauto() {
        let tensor = Tensor::<TestBackend, 1, Int>::arange_devauto(2..5);
        assert_eq!(tensor.into_data(), Data::from([2, 3, 4]));
    }

    #[test]
    fn test_arange() {
        let device = <TestBackend as Backend>::Device::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(2..5, &device);
        assert_eq!(tensor.clone().into_data(), Data::from([2, 3, 4]));
        assert_eq!(tensor.device(), device);
    }
}
