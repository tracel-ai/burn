#[burn_tensor_testgen::testgen(quantize)]

mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use burn_tensor::quantization::{
        AffineQuantization, BlockLayout, QParams, QuantizationMode, QuantizationParameters,
        QuantizationScheme, QuantizationStrategy, QuantizationType, QuantizedBytes,
        SymmetricQuantization,
    };
    use burn_tensor::{DType, Tensor, TensorData};

    // NOTE: we mark the per-block tests as `might_panic` since backends are not strictly
    // required to support this quantization scheme.
    // Also std feature gated (until `catch_unwind` is stable in core).
    #[cfg(feature = "std")]
    use burn_tensor::might_panic;

    fn get_q_params(data: TensorData) -> QParams<Vec<f32>, Vec<i8>> {
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

        let x_q = tensor.clone().quantize(&scheme, qparams);

        let x_q_data = x_q.to_data();
        let expected = TensorData::quantized(
            vec![-128i8, -39, 72, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.009_019_608, 72)),
        );

        // Values equality
        x_q_data.assert_eq(&expected, true);

        // Quantization parameters check
        let qparams = get_q_params(x_q_data);
        let expected = get_q_params(expected);
        assert_eq!(qparams.scale.len(), 1);
        assert_eq!(qparams.scale, expected.scale);
        assert_eq!(qparams.offset.as_ref().map(|x| x.len()), Some(1));
        assert_eq!(qparams.offset, expected.offset);

        // Dequantize
        let x = x_q.dequantize();

        // Precision 2 for dequantization errors
        x.into_data().assert_approx_eq(&tensor.into_data(), 2);
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

        let x_q = tensor.clone().quantize(&scheme, qparams);

        let x_q_data = x_q.to_data();
        let expected = TensorData::quantized(
            vec![-127i8, -71, 0, 35],
            [4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                0.014_173_228,
            )),
        );

        // Values equality
        x_q_data.assert_eq(&expected, true);

        // Quantization parameters check
        let qparams = get_q_params(x_q_data);
        let expected = get_q_params(expected);
        assert_eq!(qparams.scale.len(), 1);
        assert_eq!(qparams.scale, expected.scale);
        assert_eq!(qparams.offset, None);
        assert_eq!(qparams.offset, expected.offset);

        // Dequantize
        let x = x_q.dequantize();

        // Precision 2 for dequantization errors
        x.into_data().assert_approx_eq(&tensor.into_data(), 2);
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

    #[cfg(feature = "std")]
    #[might_panic(reason = "Per-block quantization is not supported")]
    #[test]
    fn should_support_quantize_per_block_symmetric_int8() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_floats(
            [
                [-1.8, -1.0, 0.0, 0.5],
                [-0.8, 1.2, 0.25, 0.5],
                [-0.08, 0.12, 0.025, 0.05],
                [0.2, 0.3, 0.4, 0.5],
                [0.1, 0.3, 0.2, 0.6],
                [4.0, 3.0, 2.0, 1.0],
                [0.4, 0.3, 0.2, 0.1],
                [0.5, 0.0, -1.0, -1.8],
            ],
            &device,
        );
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Symmetric,
            QuantizationType::QInt8,
            BlockLayout::Flat(4),
        );

        // Per-block qparams
        let scales: [f32; 8] = [
            0.014173228,
            0.009448819,
            0.0009448819,
            0.003937008,
            0.0047244094,
            0.031496063,
            0.0031496063,
            0.014173228,
        ];
        let qparams = QuantizationParameters {
            scale: Tensor::from_floats(scales, &device),
            offset: None,
        };

        let x_q = tensor.clone().quantize(&scheme, qparams);

        let x_q_data = x_q.to_data();
        let expected = TensorData::quantized(
            vec![
                [-127i8, -71, 0, 35],
                [-85, 127, 26, 53],
                [-85, 127, 26, 53],
                [51, 76, 102, 127],
                [21, 64, 42, 127],
                [127, 95, 64, 32],
                [127, 95, 64, 32],
                [35, 0, -71, -127],
            ]
            .concat(),
            [8, 4],
            QuantizationStrategy::PerBlockSymmetricInt8(
                scales
                    .iter()
                    .map(|&s| SymmetricQuantization::init(s))
                    .collect(),
                BlockLayout::Flat(4),
            ),
        );

        // Values equality
        x_q_data.assert_eq(&expected, true);

        // Quantization parameters check
        let qparams = get_q_params(x_q_data);
        let expected = get_q_params(expected);
        assert_eq!(qparams.scale.len(), 8);
        assert_eq!(qparams.scale, expected.scale);
        assert_eq!(qparams.offset, None);
        assert_eq!(qparams.offset, expected.offset);

        // Dequantize
        let x = x_q.dequantize();

        // Precision 2 for dequantization errors
        x.into_data().assert_approx_eq(&tensor.into_data(), 2);
    }

    #[allow(clippy::excessive_precision)]
    #[cfg(feature = "std")]
    #[might_panic(reason = "Per-block quantization is not supported")]
    #[test]
    fn should_support_quantize_per_block_affine_int8() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_floats(
            [
                [-1.8, -1.0, 0.0, 0.5, -0.8, 1.2, 0.25, 0.5],
                [-8., 12., 2.5, 5., 0.2, 0.3, 0.4, 0.5],
            ],
            &device,
        );
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Affine,
            QuantizationType::QInt8,
            BlockLayout::Flat(4),
        );

        // Per-block qparams
        let scales: [f32; 4] = [0.009019608, 0.007843138, 0.078431366, 0.0019607844];
        let offsets: [i8; 4] = [71, -26, -26, -128];
        let qparams = QuantizationParameters {
            scale: Tensor::from_floats(scales, &device),
            offset: Some(Tensor::from_ints(offsets, &device)),
        };

        let x_q = tensor.clone().quantize(&scheme, qparams);

        let x_q_data = x_q.to_data();
        let expected = TensorData::quantized(
            vec![
                [-128i8, -40, 71, 126],
                [-128, 127, 6, 38],
                [-128, 127, 6, 38],
                [-26, 25, 76, 127],
            ]
            .concat(),
            [2, 8],
            QuantizationStrategy::PerBlockAffineInt8(
                scales
                    .iter()
                    .zip(offsets.iter())
                    .map(|(&s, &o)| AffineQuantization::init(s, o))
                    .collect(),
                BlockLayout::Flat(4),
            ),
        );

        // Values equality
        x_q_data.assert_eq(&expected, true);

        // Quantization parameters check
        let qparams = get_q_params(x_q_data);
        let expected = get_q_params(expected);
        assert_eq!(qparams.scale.len(), 4);
        assert_eq!(qparams.scale, expected.scale);
        assert_eq!(qparams.offset.as_ref().unwrap().len(), 4);
        assert_eq!(qparams.offset, expected.offset);

        // Dequantize
        let x = x_q.dequantize();

        // Precision 2 for dequantization errors
        x.into_data().assert_approx_eq(&tensor.into_data(), 2);
    }

    #[cfg(feature = "std")]
    #[might_panic(reason = "Per-block quantization is not supported")]
    #[test]
    fn should_support_quantize_per_block_grid_symmetric_int8() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_floats(
            [
                // 2x2 blocks: [[-1.8, -1.0, 0.0, 0.5], [-0.8, 1.2, 0.25, 0.5]]
                [-1.8, -1.0, -0.8, 1.2],
                [0.0, 0.5, 0.25, 0.5],
                // 2x2 blocks: [[-0.8, 1.2, 0.25, 0.5], [0.2, 0.3, 0.4, 0.5]]
                [-0.8, 1.2, 0.2, 0.3],
                [0.25, 0.5, 0.4, 0.5],
                // 2x2 blocks: [[0.1, 0.3, 0.2, 0.6], [4.0, 3.0, 2.0, 1.0]]
                [0.1, 0.3, 4.0, 3.0],
                [0.2, 0.6, 2.0, 1.0],
                // 2x2 blocks: [[0.4, 0.3, 0.2, 0.1], [0.5, 0.0, -1.0, -1.8]]
                [0.4, 0.3, 0.5, 0.0],
                [0.2, 0.1, -1.0, -1.8],
            ],
            &device,
        );
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Symmetric,
            QuantizationType::QInt8,
            BlockLayout::Grid(2, 2),
        );

        // Per-block qparams
        let scales: [f32; 8] = [
            0.014173228,
            0.009448819,
            0.009448819,
            0.003937008,
            0.0047244094,
            0.031496063,
            0.0031496063,
            0.014173228,
        ];
        let qparams = QuantizationParameters {
            scale: Tensor::from_floats(scales, &device),
            offset: None,
        };

        let x_q = tensor.clone().quantize(&scheme, qparams);

        let x_q_data = x_q.to_data();
        let expected = TensorData::quantized(
            vec![
                [-127i8, -71, -85, 127],
                [0, 35, 26, 53],
                [-85, 127, 51, 76],
                [26, 53, 102, 127],
                [21, 64, 127, 95],
                [42, 127, 64, 32],
                [127, 95, 35, 0],
                [64, 32, -71, -127],
            ]
            .concat(),
            [8, 4],
            QuantizationStrategy::PerBlockSymmetricInt8(
                scales
                    .iter()
                    .map(|&s| SymmetricQuantization::init(s))
                    .collect(),
                BlockLayout::Grid(2, 2),
            ),
        );

        // Values equality
        x_q_data.assert_eq(&expected, true);

        // Quantization parameters check
        let qparams = get_q_params(x_q_data);
        let expected = get_q_params(expected);
        assert_eq!(qparams.scale.len(), 8);
        assert_eq!(qparams.scale, expected.scale);
        assert_eq!(qparams.offset, None);
        assert_eq!(qparams.offset, expected.offset);

        // Dequantize
        let x = x_q.dequantize();

        // Precision 2 for dequantization errors
        x.into_data().assert_approx_eq(&tensor.into_data(), 2);
    }
}
