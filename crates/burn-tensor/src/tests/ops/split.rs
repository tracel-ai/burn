#[burn_tensor_testgen::testgen(split)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{Int, Shape, Tensor, TensorData};

    #[test]
    fn test_split_evenly_divisible() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> =
            Tensor::arange(0..12, &Default::default()).split(2, 0);
        assert_eq!(tensors.len(), 3);
    }
}
