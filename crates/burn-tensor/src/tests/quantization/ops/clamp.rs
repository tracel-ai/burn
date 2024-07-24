#[burn_tensor_testgen::testgen(q_clamp)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn clamp_min() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.clamp_min(2.0);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]]), 1);
    }

    #[test]
    fn clamp_max() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.clamp_max(2.0);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[0.0, 1.0, 2.0], [2.0, 2.0, 2.0]]), 1);
    }

    #[test]
    fn clamp_min_max() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.clamp(1.0, 4.0);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[1.0, 1.0, 2.0], [3.0, 4.0, 4.0]]), 1);
    }
}
