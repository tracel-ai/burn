#[burn_tensor_testgen::testgen(q_any)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_any() {
        // Quantized [[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]
        let data = TensorData::quantized(
            vec![0i8, 0, 0, 127, -127, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.007874016)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let data_actual = tensor.any().into_data();
        let data_expected = TensorData::from([true]);
        assert_eq!(data_expected, data_actual);

        // Quantized [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        let data = TensorData::quantized(
            vec![0i8, 0, 0, 0, 0, 0],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.007874016)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let data_actual = tensor.any().into_data();
        let data_expected = TensorData::from([false]);
        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn test_any_dim() {
        // Quantized [[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]
        let data = TensorData::quantized(
            vec![0i8, 0, 0, 127, -127, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.007874016)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let data_actual = tensor.any_dim(1).into_data();
        let data_expected = TensorData::from([[false], [true]]);
        assert_eq!(data_expected, data_actual);
    }
}
