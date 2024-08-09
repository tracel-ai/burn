#[burn_tensor_testgen::testgen(q_remainder)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{
        AffineQuantization, QuantizationStrategy, SymmetricQuantization,
    };
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_remainder_basic() {
        // Quantized [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![-128i8, -85, -43, 43, 85, 127],
            [6],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.023529412, 0)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.remainder_scalar(2.0);
        let expected = TensorData::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_remainder_float() {
        // Quantized [1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![-77i8, -26, 25, 76, 127],
            [5],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.remainder_scalar(-1.5);
        let expected = TensorData::from([-0.5, -1.0, 0.0, -0.5, -1.0]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_be_zero() {
        // Quantized [0.0, 0.0, 0.0]
        let data = TensorData::quantized(
            vec![0i8, 0, 0],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.1)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.remainder_scalar(3.5);
        let expected = TensorData::from([0.0, 0.0, 0.0]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }

    #[test]
    fn should_have_no_remainder() {
        // Quantized [-4.0, 4.0]
        let data = TensorData::quantized(
            vec![-127i8, 127],
            [2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.031496063)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.remainder_scalar(4.0);
        let expected = TensorData::from([-0.0, 0.0]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }

    #[test]
    fn should_be_negative() {
        // Quantized [-7.0, -3.0, 2.0, 6.0]
        let data = TensorData::quantized(
            vec![-128i8, -50, 48, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.050980393, 9)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.remainder_scalar(-2.5);
        let expected = TensorData::from([-2.0, -0.50, -0.50, -1.5]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_fp_dividends() {
        // Quantized [-7.5, -2.5, 2.5, 7.5]
        let data = TensorData::quantized(
            vec![-127i8, -42, 42, 127],
            [4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05905512)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.remainder_scalar(3.0);
        let expected = TensorData::from([1.5, 0.5, 2.5, 1.5]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_large_divisor() {
        // Quantized [-1.0, 1.0]
        let data = TensorData::quantized(
            vec![-127i8, 127],
            [2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.007874016)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.remainder_scalar(10.0);
        let expected = TensorData::from([9.0, 1.0]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_remainder_op() {
        // Quantized [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![-128i8, -85, -43, 43, 85, 127],
            [6],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.023529412, 0)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor % 2.0;
        let expected = TensorData::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
