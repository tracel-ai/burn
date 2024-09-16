#[burn_tensor_testgen::testgen(quantize)]
mod tests {
    use super::*;
    use burn_tensor::ops::QTensorOps;
    use burn_tensor::quantization::{
        AffineQuantization, QuantizationParameters, QuantizationScheme, QuantizationStrategy,
        QuantizationType, SymmetricQuantization,
    };
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_quantize_affine_int8() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_floats([-1.8, -1.0, 0.0, 0.5], &device);
        let scheme = QuantizationScheme::PerTensorAffine(QuantizationType::QInt8);
        let qparams = QuantizationParameters {
            scale: Tensor::from_floats([0.009_019_608], &device),
            offset: Some(Tensor::from_ints([72], &device)),
        };

        let x_q = tensor.quantize(&scheme, qparams);

        let expected = TensorData::quantized(
            vec![-128i8, -39, 72, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.009_019_608, 72)),
        );

        x_q.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn should_support_quantize_symmetric_int8() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_floats([-1.8, -1.0, 0.0, 0.5], &device);
        let scheme = QuantizationScheme::PerTensorSymmetric(QuantizationType::QInt8);
        let qparams = QuantizationParameters {
            scale: Tensor::from_floats([0.014_173_228], &device),
            offset: None,
        };

        let x_q = tensor.quantize(&scheme, qparams);

        let expected = TensorData::quantized(
            vec![-127i8, -71, 0, 35],
            [4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                0.014_173_228,
            )),
        );

        x_q.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn should_support_dequantize() {
        let device = Default::default();
        // Quantized [-1.8, -1.0, 0.0, 0.5]
        let data = TensorData::quantized(
            vec![-127i8, -71, 0, 35],
            [4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                0.014_173_228,
            )),
        );
        let x_q = Tensor::<TestBackend, 1>::from_data(data, &device);

        let x = x_q.dequantize();

        // Precision 2 for dequantization errors
        x.to_data()
            .assert_approx_eq(&TensorData::from([-1.8, -1.0, 0.0, 0.5]), 2);
    }

    #[test]
    fn should_support_quantize_dynamic_int8() {
        let device = Default::default();
        // NOTE: we use fully representable values since different backend implementations could differ slightly
        // due to rounding discrepancies
        let tensor = Tensor::<TestBackend, 1>::from_floats([5., 0., 4., -10.], &device);
        let scheme = QuantizationScheme::PerTensorAffine(QuantizationType::QInt8);

        let x_q = tensor.quantize_dynamic(&scheme);

        let expected = TensorData::quantized(
            vec![127i8, 42, 110, -128],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, 42)),
        );

        x_q.to_data().assert_eq(&expected, false);
    }
}
