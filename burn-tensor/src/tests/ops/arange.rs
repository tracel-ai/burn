#[burn_tensor_testgen::testgen(arange)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn test_arange() {
        let device = <TestBackend as Backend>::Device::default();

        let tensor = Tensor::<TestBackend, 1, Int>::arange(2..5, &device);
        assert_eq!(tensor.into_data(), Data::from([2, 3, 4]));

        // Test arange with negative numbers
        let tensor = Tensor::<TestBackend, 1, Int>::arange(-10..-5, &device);
        assert_eq!(tensor.into_data(), Data::from([-10, -9, -8, -7, -6]));

        let tensor = Tensor::<TestBackend, 1, Int>::arange(-3..0, &device);
        assert_eq!(tensor.into_data(), Data::from([-3, -2, -1]));

        // Test arange with a mix of positive and negative numbers
        let tensor = Tensor::<TestBackend, 1, Int>::arange(-2..3, &device);
        assert_eq!(tensor.clone().into_data(), Data::from([-2, -1, 0, 1, 2]));
        assert_eq!(tensor.device(), device);
    }
}
