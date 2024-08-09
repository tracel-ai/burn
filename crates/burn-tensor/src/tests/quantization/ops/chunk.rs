#[burn_tensor_testgen::testgen(q_chunk)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    fn test_chunk_evenly_divisible() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [6],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let tensors: Vec<Tensor<TestBackend, 1>> = tensor.chunk(3, 0);
        assert_eq!(tensors.len(), 3);

        let expected = vec![
            TensorData::from([0., 1.]),
            TensorData::from([2., 3.]),
            TensorData::from([4., 5.]),
        ];

        // Precision 1 to approximate de/quantization errors
        for (index, tensor) in tensors.into_iter().enumerate() {
            tensor
                .dequantize()
                .to_data()
                .assert_approx_eq(&expected[index], 1);
        }
    }

    #[test]
    fn test_chunk_not_evenly_divisible() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let data = TensorData::quantized(
            vec![0i8, 21, 42, 64, 85, 106, 127],
            [7],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let tensors: Vec<Tensor<TestBackend, 1>> = tensor.chunk(4, 0);
        assert_eq!(tensors.len(), 4);

        let expected = vec![
            TensorData::from([0., 1.]),
            TensorData::from([2., 3.]),
            TensorData::from([4., 5.]),
            TensorData::from([6.]),
        ];

        // Precision 1 to approximate de/quantization errors
        for (index, tensor) in tensors.into_iter().enumerate() {
            tensor
                .dequantize()
                .to_data()
                .assert_approx_eq(&expected[index], 1);
        }
    }

    #[test]
    fn test_chunk_not_divisible() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [6],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let tensors: Vec<Tensor<TestBackend, 1>> = tensor.chunk(7, 0);
        assert_eq!(tensors.len(), 6);

        let expected = vec![
            TensorData::from([0.]),
            TensorData::from([1.]),
            TensorData::from([2.]),
            TensorData::from([3.]),
            TensorData::from([4.]),
            TensorData::from([5.]),
        ];

        // Precision 1 to approximate de/quantization errors
        for (index, tensor) in tensors.into_iter().enumerate() {
            tensor
                .dequantize()
                .to_data()
                .assert_approx_eq(&expected[index], 1);
        }
    }

    #[test]
    fn test_chunk_multi_dimension() {
        // Quantized [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [1, 6],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let tensors: Vec<Tensor<TestBackend, 2>> = tensor.chunk(2, 1);
        assert_eq!(tensors.len(), 2);

        let expected = vec![
            TensorData::from([[0., 1., 2.]]),
            TensorData::from([[3., 4., 5.]]),
        ];

        // Precision 1 to approximate de/quantization errors
        for (index, tensor) in tensors.into_iter().enumerate() {
            tensor
                .dequantize()
                .to_data()
                .assert_approx_eq(&expected[index], 1);
        }
    }

    #[test]
    #[should_panic]
    fn test_invalid_dim() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [6],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensors = TestTensor::<1>::from_data(data, &Default::default()).chunk(6, 1);
    }
}
