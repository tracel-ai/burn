#[burn_tensor_testgen::testgen(q_split)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_split_evenly_divisible() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [6],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let tensors = tensor.split(2, 0);
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
    fn test_split_not_evenly_divisible() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let data = TensorData::quantized(
            vec![0i8, 21, 42, 64, 85, 106, 127],
            [7],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let tensors = tensor.split(2, 0);
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
    fn test_split_along_dim1() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let tensors = tensor.split(2, 1);
        assert_eq!(tensors.len(), 2);

        let expected = vec![
            TensorData::from([[0., 1.], [3., 4.]]),
            TensorData::from([[2.], [5.]]),
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
    fn test_split_split_size_larger_than_tensor_size() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [6],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let tensors = tensor.split(10, 0);
        assert_eq!(tensors.len(), 1);

        let expected = vec![TensorData::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])];

        // Precision 1 to approximate de/quantization errors
        for (index, tensor) in tensors.into_iter().enumerate() {
            tensor
                .dequantize()
                .to_data()
                .assert_approx_eq(&expected[index], 1);
        }
    }

    #[test]
    #[should_panic(
        expected = "split_size must be greater than 0 unless the tensor size along the dimension is 0."
    )]
    fn test_split_with_zero_split_size_non_zero_tensor() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [6],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let _ = tensor.split(0, 0);
    }

    #[test]
    #[should_panic(expected = "Given dimension is greater than or equal to the tensor rank.")]
    fn test_split_invalid_dim() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [6],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let _ = tensor.split(1, 2);
    }

    #[test]
    fn test_split_with_sizes() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [6],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let tensors = tensor.split_with_sizes(vec![2, 3, 1], 0);
        assert_eq!(tensors.len(), 3);

        let expected = vec![
            TensorData::from([0., 1.]),
            TensorData::from([2., 3., 4.]),
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
    #[should_panic(
        expected = "The sum of split_sizes must equal the tensor size along the specified dimension."
    )]
    fn test_split_with_sizes_invalid_sum() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [6],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let _ = tensor.split_with_sizes(vec![2, 2, 1], 0);
    }

    #[test]
    fn test_split_with_sizes_zero_length() {
        // Quantized [0.0, 2.0, 5.0]
        let data = TensorData::quantized(
            vec![0i8, 51, 127],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let tensors = tensor.split_with_sizes(vec![0, 1, 2], 0);
        assert_eq!(tensors.len(), 2);

        let expected = vec![TensorData::from([0.]), TensorData::from([2., 5.])];

        // Precision 1 to approximate de/quantization errors
        for (index, tensor) in tensors.into_iter().enumerate() {
            tensor
                .dequantize()
                .to_data()
                .assert_approx_eq(&expected[index], 1);
        }
    }
}
