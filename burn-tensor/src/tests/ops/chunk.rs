#[burn_tensor_testgen::testgen(chunk)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{Data, Int, Shape, Tensor};

    fn test_chunk_evenly_divisible() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> = Tensor::arange(0..12).chunk(6, 0);
        assert_eq!(tensors.len(), 6);

        let expected = vec![
            Data::from([0, 1]),
            Data::from([2, 3]),
            Data::from([4, 5]),
            Data::from([6, 7]),
            Data::from([8, 9]),
            Data::from([10, 11]),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.to_data(), expected[index]);
        }
    }

    #[test]
    fn test_chunk_not_evenly_divisible() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> = Tensor::arange(0..11).chunk(6, 0);
        assert_eq!(tensors.len(), 6);

        let expected = vec![
            Data::from([0, 1]),
            Data::from([2, 3]),
            Data::from([4, 5]),
            Data::from([6, 7]),
            Data::from([8, 9]),
            Data::from([10]),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.to_data(), expected[index]);
        }
    }

    #[test]
    fn test_chunk_not_evenly_divisible_remains_several() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> = Tensor::arange(0..100).chunk(8, 0);
        assert_eq!(tensors.len(), 8);

        let expected = [13, 13, 13, 13, 13, 13, 13, 9];

        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.shape().dims[0], expected[index]);
        }
    }

    #[test]
    fn test_chunk_not_divisible() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> = Tensor::arange(0..6).chunk(7, 0);
        assert_eq!(tensors.len(), 6);

        let expected = vec![
            Data::from([0]),
            Data::from([1]),
            Data::from([2]),
            Data::from([3]),
            Data::from([4]),
            Data::from([5]),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.to_data(), expected[index]);
        }
    }

    #[test]
    fn test_chunk_multi_dimension() {
        let tensors: Vec<Tensor<TestBackend, 2, Int>> =
            Tensor::from_data(Data::from([[0, 1, 2, 3]])).chunk(2, 1);
        assert_eq!(tensors.len(), 2);

        let expected = vec![Data::from([[0, 1]]), Data::from([[2, 3]])];

        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.to_data(), expected[index]);
        }
    }

    #[test]
    #[should_panic]
    fn test_invalid_dim() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> = Tensor::arange(0..12).chunk(6, 1);
    }
}
