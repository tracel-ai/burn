#[burn_tensor_testgen::testgen(q_sin)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_sin_ops() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.sin();
        let expected = TensorData::from([[0.0, 0.8414, 0.9092], [0.1411, -0.7568, -0.9589]]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }
}
