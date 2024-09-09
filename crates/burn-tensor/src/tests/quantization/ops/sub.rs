#[burn_tensor_testgen::testgen(q_sub)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{backend::Backend, Tensor, TensorData};

    #[test]
    fn should_support_sub_ops() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        let data = TensorData::quantized(
            vec![11i8, 34, 57, 81, 104, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.043137256, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output = tensor_1 - tensor_2;
        let expected = TensorData::from([[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn test_sub_broadcast() {
        let data_1 = TensorData::from([[0.0, 1.0, 2.0]]);
        let data_2 = TensorData::from([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0]]
        let data = TensorData::quantized(
            vec![-128i8, -1, 127],
            [1, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.007843138, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
        let data = TensorData::quantized(
            vec![-32i8, -1, 31, 63, 95, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.03137255, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output = tensor_1 - tensor_2;
        let expected = TensorData::from([[-3.0, -3.0, -3.0], [-6.0, -6.0, -6.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_sub_scalar_ops() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());
        let scalar = 2.0;

        let output = tensor - scalar;
        let expected = TensorData::from([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }
}
