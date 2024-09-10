#[burn_tensor_testgen::testgen(q_recip)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_recip_ops() {
        // Quantized [[0.5, 1.0, 2.0], [3.0, -4.0, -5.0]]
        let data = TensorData::quantized(
            vec![47i8, 63, 95, 127, -96, -128],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.03137255, 31)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.recip();
        let expected = TensorData::from([[2.0, 1.0, 0.5], [0.33333, -0.25, -0.2]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
