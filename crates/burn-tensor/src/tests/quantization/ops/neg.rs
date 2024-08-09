#[burn_tensor_testgen::testgen(q_neg)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_neg_ops() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        let output = tensor.neg();
        let expected = TensorData::from([[-0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]).convert::<f32>();

        // -0.0 is represented differently than 0.0 so we make sure the values are the same in f32
        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
