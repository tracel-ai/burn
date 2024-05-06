#[burn_tensor_testgen::testgen(indices)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Data, Int, Shape, Tensor};

    #[test]
    fn test_arange() {
        let device = <TestBackend as Backend>::Device::default();

        // Test a single element tensor
        let tensor = Tensor::<TestBackend, 1, Int>::indices::<2>(Shape { dims: [1] }, &device);
        assert_eq!(tensor.into_data(), Data::from([[0]]));

        // Test for a 2x2 tensor
        let tensor = Tensor::<TestBackend, 2, Int>::indices::<3>(Shape { dims: [2, 2] }, &device);
        assert_eq!(
            tensor.into_data(),
            Data::from([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
        );
    }
}
