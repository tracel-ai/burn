#[burn_tensor_testgen::testgen(quantize)]
mod tests {
    use super::*;
    use burn_tensor::ops::QTensorOps;
    use burn_tensor::quantization::{
        AffineQuantization, QParams, QuantizationMode, QuantizationParameters, QuantizationScheme,
        QuantizationStrategy, QuantizationType, QuantizedBytes, SymmetricQuantization,
    };
    use burn_tensor::{DType, Tensor, TensorData};

    fn get_q_params(data: TensorData) -> QParams<f32, i8> {
        let num_elements = data.num_elements();
        let scheme = if let DType::QFloat(scheme) = data.dtype {
            scheme
        } else {
            unreachable!()
        };
        let q_bytes = QuantizedBytes {
            bytes: data.into_bytes(),
            scheme,
            num_elements,
        };
        q_bytes.into_vec_i8().1
    }

    #[test]
    fn should_support_quantize_affine_int8() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_floats([-1.8, -1.0, 0.0, 0.5], &device);
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8);
        let qparams = QuantizationParameters {
            scale: Tensor::from_floats([0.009_019_608], &device),
            offset: Some(Tensor::from_ints([72], &device)),
        };

        let x_q = tensor.quantize(&scheme, qparams).into_data();

        let expected = TensorData::quantized(
            vec![-128i8, -39, 72, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.009_019_608, 72)),
        );

        // Values equality
        x_q.assert_eq(&expected, true);

        // Quantization parameters check
        let qparams = get_q_params(x_q);
        let expected = get_q_params(expected);
        assert_eq!(qparams.scale, expected.scale);
        assert_eq!(qparams.offset, expected.offset);
    }

    #[test]
    fn should_support_quantize_symmetric_int8() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_floats([-1.8, -1.0, 0.0, 0.5], &device);
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8);
        let qparams = QuantizationParameters {
            scale: Tensor::from_floats([0.014_173_228], &device),
            offset: None,
        };

        let x_q = tensor.quantize(&scheme, qparams).into_data();

        let expected = TensorData::quantized(
            vec![-127i8, -71, 0, 35],
            [4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                0.014_173_228,
            )),
        );

        // Values equality
        x_q.assert_eq(&expected, true);

        // Quantization parameters check
        let qparams = get_q_params(x_q);
        let expected = get_q_params(expected);
        assert_eq!(qparams.scale, expected.scale);
        assert_eq!(qparams.offset, expected.offset);
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
        let x_q = TestTensor::<1>::from_data(data, &device);

        let x = x_q.dequantize();

        // Precision 2 for dequantization errors
        x.into_data()
            .assert_approx_eq(&TensorData::from([-1.8, -1.0, 0.0, 0.5]), 2);
    }

    #[test]
    fn should_support_quantize_dynamic_int8() {
        let device = Default::default();
        // NOTE: we use fully representable values since different backend implementations could differ slightly
        // due to rounding discrepancies
        let tensor = TestTensor::<1>::from_floats([5., 0., 4., -10.], &device);
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8);

        let x_q = tensor.quantize_dynamic(&scheme);

        let expected = TensorData::quantized(
            vec![127i8, 42, 110, -128],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, 42)),
        );

        x_q.into_data().assert_eq(&expected, false);
    }
}
