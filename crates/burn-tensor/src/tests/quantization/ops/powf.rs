#[burn_tensor_testgen::testgen(q_powf)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{
        AffineQuantization, QuantizationStrategy, SymmetricQuantization,
    };
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_powf_ops() {
        let device = Default::default();
        // Quantized [[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-77i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        // Quantized [[1.0, 1.0, 2.0], [3.0, 4.0, 2.0]] (with range [1., 5.] to reduce quantization errors)
        let data = TensorData::quantized(
            vec![-77i8, -77, -26, 25, 76, -26],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_pow = TestTensor::<2>::from_data(data, &device);

        let output = tensor.powf(tensor_pow);
        let expected = TensorData::from([[1.0, 1.0, 4.0], [27.0, 256.0, 25.0]]);

        // NOTE: we set higher tolerance (0.2) due to larger de/quantization errors accumulation w/ powers
        output
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.2);
    }

    #[test]
    fn should_support_neg_power() {
        let device = Default::default();
        // Quantized [[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-77i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        // Quantized [[-0.95, -0.67, -0.45], [-0.24, -0.5, -0.6]]
        let data = TensorData::quantized(
            vec![-128i8, -53, 6, 63, -7, -34],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.00372549, 127)),
        );
        let tensor_pow = TestTensor::<2>::from_data(data, &device);

        let output = tensor.powf(tensor_pow);
        let expected = TensorData::from([[1., 1., 0.73204285], [0.76822936, 0.5, 0.38073079]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_neg_values_with_even_power() {
        let device = Default::default();
        // Quantized [[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]
        let data = TensorData::quantized(
            vec![126i8, 75, 24, -27, -78, -128],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, 126)),
        );
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &device);
        // Quantized [[4.0, 2.0, 4.0], [2.0, 4.0, 2.0]] (with range [2., 5.] to reduce quantization errors)
        let data = TensorData::quantized(
            vec![76i8, -26, 76, -26, 76, -26],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_pow = TestTensor::<2>::from_data(data, &device);

        let output = tensor.powf(tensor_pow);
        let expected = TensorData::from([[0.0, 1.0, 16.0], [9.0, 256.0, 25.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_neg_values_with_odd_power() {
        let device = Default::default();
        // Quantized [[0.0, -1.0, -2.0], [-3.0, -4.0, -4.0]] (with range [-5., 0.] to reduce quantization errors)
        let data = TensorData::quantized(
            vec![126i8, 75, 24, -27, -78, -78],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, 126)),
        );
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());
        // Quantized [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
        let data = TensorData::quantized(
            vec![127i8, 127, 127, 127, 127, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor_pow = TestTensor::<2>::from_data(data, &device);

        let output = tensor.powf(tensor_pow);
        let expected = TensorData::from([[0.0, -1.0, -8.0], [-27.0, -64.0, -64.0]]);

        // NOTE: we set higher tolerance (0.3) due to larger de/quantization errors accumulation w/ powers
        // and large output range
        output
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.3);
    }
}
