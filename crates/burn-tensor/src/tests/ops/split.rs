#[burn_tensor_testgen::testgen(split)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{Int, Shape, Tensor, TensorData};

    #[test]
    fn test_split_evenly_divisible() {
        let device = Default::default();
        let tensors =
            TestTensor::<2>::from_data([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], &device);

        let split_tensors = tensors.split(2, 0);
        assert_eq!(split_tensors.len(), 3);

        let expected = vec![
            TensorData::from([[0, 1], [2, 3]]),
            TensorData::from([[4, 5], [6, 7]]),
            TensorData::from([[8, 9], [10, 11]]),
        ];

        for (index, tensor) in split_tensors.iter().enumerate() {
            tensor.to_data().assert_eq(&expected[index], false);
        }
    }

    #[test]
    fn test_split_not_evenly_divisible() {
        let device = Default::default();
        let tensors = TestTensor::<2>::from_data([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], &device);

        let split_tensors = tensors.split(2, 0);
        assert_eq!(split_tensors.len(), 3);

        let expected = vec![
            TensorData::from([[0, 1], [2, 3]]),
            TensorData::from([[4, 5], [6, 7]]),
            TensorData::from([[8, 9]]),
        ];

        for (index, tensor) in split_tensors.iter().enumerate() {
            tensor.to_data().assert_eq(&expected[index], false);
        }
    }
}
