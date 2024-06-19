#[burn_tensor_testgen::testgen(chunk)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::{backend::Backend, Int, Shape, Tensor, TensorData};

    fn test_chunk_evenly_divisible() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> =
            Tensor::arange(0..12, &Default::default()).chunk(6, 0);
        assert_eq!(tensors.len(), 6);

        let expected = vec![
            TensorData::from([0, 1]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([2, 3]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([4, 5]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([6, 7]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([8, 9]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([10, 11]).convert::<<TestBackend as Backend>::IntElem>(),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            tensor.to_data().assert_eq(&expected[index], true);
        }
    }

    #[test]
    fn test_chunk_not_evenly_divisible() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> =
            Tensor::arange(0..11, &Default::default()).chunk(6, 0);
        assert_eq!(tensors.len(), 6);

        let expected = vec![
            TensorData::from([0, 1]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([2, 3]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([4, 5]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([6, 7]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([8, 9]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([10]).convert::<<TestBackend as Backend>::IntElem>(),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            tensor.to_data().assert_eq(&expected[index], true);
        }
    }

    #[test]
    fn test_chunk_not_evenly_divisible_remains_several() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> =
            Tensor::arange(0..100, &Default::default()).chunk(8, 0);
        assert_eq!(tensors.len(), 8);

        let expected = [13, 13, 13, 13, 13, 13, 13, 9];

        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.shape().dims[0], expected[index]);
        }
    }

    #[test]
    fn test_chunk_not_divisible() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> =
            Tensor::arange(0..6, &Default::default()).chunk(7, 0);
        assert_eq!(tensors.len(), 6);

        let expected = vec![
            TensorData::from([0]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([1]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([2]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([3]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([4]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([5]).convert::<<TestBackend as Backend>::IntElem>(),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            tensor.to_data().assert_eq(&expected[index], true);
        }
    }

    #[test]
    fn test_chunk_multi_dimension() {
        let tensors: Vec<Tensor<TestBackend, 2, Int>> =
            Tensor::from_data(TensorData::from([[0, 1, 2, 3]]), &Default::default()).chunk(2, 1);
        assert_eq!(tensors.len(), 2);

        let expected = vec![
            TensorData::from([[0, 1]]).convert::<<TestBackend as Backend>::IntElem>(),
            TensorData::from([[2, 3]]).convert::<<TestBackend as Backend>::IntElem>(),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            tensor.to_data().assert_eq(&expected[index], true);
        }
    }

    #[test]
    #[should_panic]
    fn test_invalid_dim() {
        let tensors: Vec<Tensor<TestBackend, 1, Int>> =
            Tensor::arange(0..12, &Default::default()).chunk(6, 1);
    }
}
