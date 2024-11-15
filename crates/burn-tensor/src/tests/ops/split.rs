#[burn_tensor_testgen::testgen(split)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{Int, Shape, Tensor, TensorData};

    #[test]
    fn test_split_evenly_divisible() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> = Tensor::arange(0..12, &Default::default())
            .reshape([5, 2])
            .split(2, 0);
        assert_eq!(tensors.len(), 3);

        let expected = vec![
            TensorData::from([[0, 1], [2, 3]]),
            TensorData::from([[4, 5], [6, 7]]),
            TensorData::from([[8, 9], [10, 11]]),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            tensor.to_data().assert_eq(&expected[index], false);
        }
    }
}
