#[burn_tensor_testgen::testgen(q_powf_scalar)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{
        AffineQuantization, QuantizationStrategy, SymmetricQuantization,
    };
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_powf_ops() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.powf_scalar(0.71);
        let expected = TensorData::from([[0.0, 1.0, 1.6358], [2.182, 2.6759, 3.1352]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_neg_power() {
        // Quantized [[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![25i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.powf_scalar(-0.33);
        let expected =
            TensorData::from([[1.0, 1.0, 0.79553646], [0.695905, 0.6328783, 0.58794934]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_neg_values_with_even_power() {
        // Quantized [[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]
        let data = TensorData::quantized(
            vec![126i8, 75, 24, -27, -78, -128],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, 126)),
        );
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.powf_scalar(2.0);
        let expected = TensorData::from([[0., 1., 4.], [9., 16., 25.]]);

        // NOTE: we set higher tolerance (0.2) due to larger de/quantization errors accumulation w/ powers
        output
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.2);
    }

    #[test]
    fn should_support_neg_values_with_odd_power() {
        // Quantized [[0.0, -1.0, -2.0], [-3.0, -4.0, -4.0]] (with range [-5., 0.] to reduce quantization errors)
        let data = TensorData::quantized(
            vec![126i8, 75, 24, -27, -78, -78],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, 126)),
        );
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.powf_scalar(3.0);
        let expected = TensorData::from([[0.0, -1.0, -8.0], [-27.0, -64.0, -64.0]]);

        // NOTE: we set higher tolerance (0.3) due to larger de/quantization errors accumulation w/ powers
        // and large output range
        output
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.3);
    }
}
