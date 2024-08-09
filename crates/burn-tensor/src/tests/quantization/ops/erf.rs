#[burn_tensor_testgen::testgen(q_erf)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_erf_ops() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.erf();
        let expected = TensorData::from([[0.0000, 0.8427, 0.9953], [1.0000, 1.0000, 1.0000]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_erf_ops_with_negative_number() {
        // Quantized [[-0.056, -0.043, -0.089], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-1i8, -1, -2, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.erf();
        let expected = TensorData::from([
            [-0.06312324, -0.048490416, -0.10016122],
            [1.0000, 1.0000, 1.0000],
        ]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
