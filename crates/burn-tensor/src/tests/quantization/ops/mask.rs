#[burn_tensor_testgen::testgen(q_mask)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Bool, Int, Tensor, TensorData};

    #[test]
    fn should_support_mask_where_ops() {
        let device = Default::default();
        // Quantized [[1.0, 7.0], [2.0, 3.0]]
        let data = TensorData::quantized(
            vec![18i8, 127, 36, 54],
            [2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::from([[true, false], [false, true]]),
            &device,
        );
        // Quantized [[1.0, 7.0], [2.0, 3.0]]
        let data = TensorData::quantized(
            vec![48i8, 74, 101, 127],
            [2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.037795275)),
        );
        let value = TestTensor::<2>::from_data(data, &device);

        let output = tensor.mask_where(mask, value);
        let expected = TensorData::from([[1.8, 7.0], [2.0, 4.8]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_mask_fill_ops() {
        let device = Default::default();
        // Quantized [[1.0, 7.0], [2.0, 3.0]]
        let data = TensorData::quantized(
            vec![18i8, 127, 36, 54],
            [2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let mask = Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::from([[true, false], [false, true]]),
            &device,
        );

        let output = tensor.mask_fill(mask, 2.0);
        let expected = TensorData::from([[2.0, 7.0], [2.0, 2.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
