#[burn_tensor_testgen::testgen(q_tanh)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_tanh_ops() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.tanh();
        let expected = TensorData::from([[0.0, 0.7615, 0.9640], [0.9950, 0.9993, 0.9999]]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }
}
