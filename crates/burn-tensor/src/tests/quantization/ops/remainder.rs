#[burn_tensor_testgen::testgen(q_remainder)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{
        AffineQuantization, QuantizationStrategy, SymmetricQuantization,
    };
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_remainder_basic() {
        let device = Default::default();
        let lhs = TestTensor::<1>::from_data(
            // [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
            TensorData::quantized(
                vec![-127i8, -85, -42, 42, 85, 127],
                [6],
                QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                    0.023622047244094488,
                )),
            ),
            &device,
        );
        let rhs = TestTensor::<1>::from_data(
            // [2.0, 3.0, 1.0, 2.0, 1.0, 3.0]
            TensorData::quantized(
                vec![85i8, 127, 42, 85, 42, 127],
                [6],
                QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                    0.023622047244094488,
                )),
            ),
            &device,
        );

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([1., 1., 0., 1., 0., 0.]);
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_remainder_basic_scalar() {
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
        let device = Default::default();
        let lhs = TestTensor::<1>::from_data(
            // [1.0, 2.0, 3.0, 4.0, 5.0]
            TensorData::quantized(
                vec![-77i8, -26, 25, 76, 127],
                [5],
                QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019623, -128)),
            ),
            &device,
        );
        let rhs = TestTensor::<1>::from_data(
            // [1.4233, 2.7313, 0.2641, 1.9651, 0.5897]
            TensorData::quantized(
                vec![5i8, 127, -103, 56, -73],
                [5],
                QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.010700, -128)),
            ),
            &device,
        );

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([1., 2., 0.0949, 0.0698, 0.2824]);
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_remainder_float_scalar() {
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
        let device = Default::default();
        let lhs = TestTensor::<1>::from_data(
            // [0.0, 0.0, 0.0]
            TensorData::quantized(
                vec![0i8, 0, 0],
                [3],
                QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.1)),
            ),
            &device,
        );
        let rhs = TestTensor::<1>::from_data(
            // [3.5, -2.1, 1e-5 rounded]
            TensorData::quantized(
                vec![127i8, -128, -33],
                [3],
                QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.021961, -32)),
            ),
            &device,
        );

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([0.0, 0.0, 0.0]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_be_zero_scalar() {
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
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_have_no_remainder() {
        // quantization errors introduces remainder values
        let device = Default::default();
        let lhs = TestTensor::<1>::from_data(
            // [-1.4843, 1.135, -2.1563, 1.0862, 0.5034, 3.6587]
            TensorData::quantized(
                vec![-52i8, 39, -75, 38, 17, 127],
                [6],
                QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                    0.028808661333219274,
                )),
            ),
            &device,
        );
        let rhs = TestTensor::<1>::from_data(
            // [1.4843, 1.135, 2.1563, 1.0862, 0.5034, 3.6587]
            TensorData::quantized(
                vec![52i8, 39, 75, 38, 17, 127],
                [6],
                QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                    0.028808661333219274,
                )),
            ),
            &device,
        );

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_have_no_remainder_scalar() {
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
        let device = Default::default();

        let lhs = TestTensor::<1>::from_data(
            // [-7.0, -3.0, 2.0, 6.0]
            TensorData::quantized(
                vec![-128i8, -50, 48, 127],
                [4],
                QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.050980, 9)),
            ),
            &device,
        );
        let rhs = TestTensor::<1>::from_data(
            // [-2.5, -2.1, -1.5, -3.25]
            TensorData::quantized(
                vec![-69i8, -38, 9, -128],
                [4],
                QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.012757, 127)),
            ),
            &device,
        );

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([-2., -0.9, -1., -0.5]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_be_negative_scalar() {
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
        let device = Default::default();

        let lhs = TestTensor::<1>::from_data(
            // [-1.0, 1.0, -1.5, 1.5, -1.0, 1.0, -1.5, 1.5]
            TensorData::quantized(
                vec![-85i8, 85, -128, 127, -85, 85, -128, 127],
                [8],
                QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011765, 0)),
            ),
            &device,
        );

        let rhs = TestTensor::<1>::from_data(
            // [10.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0]
            TensorData::quantized(
                vec![127i8, 127, 127, 127, -127, -127, -127, -127],
                [8],
                QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.078431, 0)),
            ),
            &device,
        );

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([9., 1., 8.5, 1.5, -1., -9., -1.5, -8.5]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_large_divisor_scalar() {
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
        let device = Default::default();
        let lhs = TestTensor::<1>::from_data(
            // [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
            TensorData::quantized(
                vec![-127i8, -85, -42, 42, 85, 127],
                [6],
                QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                    0.023622047244094488,
                )),
            ),
            &device,
        );
        let rhs = TestTensor::<1>::from_data(
            // [2.0, 3.0, 1.0, 2.0, 1.0, 3.0]
            TensorData::quantized(
                vec![85i8, 127, 42, 85, 42, 127],
                [6],
                QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                    0.023622047244094488,
                )),
            ),
            &device,
        );

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([1., 1., 0., 1., 0., 0.]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_remainder_scalar_op() {
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
