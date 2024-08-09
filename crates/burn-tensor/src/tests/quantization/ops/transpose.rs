#[burn_tensor_testgen::testgen(q_transpose)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_transpose_ops() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        let data = TensorData::quantized(
            vec![-128i8, -105, -82, -58, -35, -12, 11, 34, 57, 81, 104, 127],
            [2, 2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.043137256, -128)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let output = tensor.transpose();
        let expected = TensorData::from([
            [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
            [[6.0, 9.0], [7.0, 10.0], [8.0, 11.0]],
        ]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_swap_dims() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        let data = TensorData::quantized(
            vec![-128i8, -105, -82, -58, -35, -12, 11, 34, 57, 81, 104, 127],
            [2, 2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.043137256, -128)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let output = tensor.swap_dims(0, 2);
        let expected = TensorData::from([
            [[0.0, 6.0], [3.0, 9.0]],
            [[1.0, 7.0], [4.0, 10.0]],
            [[2.0, 8.0], [5.0, 11.0]],
        ]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
