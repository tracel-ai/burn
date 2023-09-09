#[burn_tensor_testgen::testgen(arange)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn test_arange() {
        let tensor = Tensor::<TestBackend, 1, Int>::arange(2..5);
        assert_eq!(tensor.into_data(), Data::from([2, 3, 4]));
    }

    #[test]
    fn test_arange_device() {
        let device = Tensor::<TestBackend, 1>::from_data(Data::from([0.0])).device();

        let tensor = Tensor::<TestBackend, 1, Int>::arange_device(2..5, &device);
        assert_eq!(tensor.clone().into_data(), Data::from([2, 3, 4]));
        assert_eq!(tensor.device(), device);
    }
}
