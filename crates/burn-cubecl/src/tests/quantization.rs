#[burn_tensor_testgen::testgen(quantization)]
mod tests {
    use super::*;
    use burn_tensor::{
        quantization::{BlockLayout, QuantizationMode, QuantizationScheme, QuantizationType},
        Tensor,
    };

    use alloc::{vec, vec::Vec};
    use burn_tensor::{
        quantization::{AffineQuantization, QParams, QuantizationStrategy, QuantizedBytes},
        DType, TensorData,
    };

    #[test]
    fn should_quantize_dequantize_symmetric_single() {
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8);
        let input = Tensor::<TestBackend, 1>::from_floats([-1.8], &Default::default());
        let input_ref =
            Tensor::<ReferenceBackend, 1>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme);

        output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output.to_data().assert_approx_eq(&output_ref.to_data(), 3);
    }

    #[test]
    fn should_quantize_dequantize_affine_single() {
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8);
        let input = Tensor::<TestBackend, 1>::from_floats([-1.8], &Default::default());
        let input_ref =
            Tensor::<ReferenceBackend, 1>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme);

        output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output.to_data().assert_approx_eq(&output_ref.to_data(), 2);
    }

    #[test]
    fn should_quantize_dequantize_symmetric_multiple() {
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8);
        let input =
            Tensor::<TestBackend, 1>::from_floats([-1.8, -1.0, 0.0, 0.5, 0.0], &Default::default());
        let input_ref =
            Tensor::<ReferenceBackend, 1>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme);

        output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output.to_data().assert_approx_eq(&output_ref.to_data(), 3);
    }

    #[test]
    fn should_quantize_dequantize_affine_multiple() {
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8);
        let input =
            Tensor::<TestBackend, 1>::from_floats([-1.8, -1.0, 0.0, 0.5, 0.0], &Default::default());
        let input_ref =
            Tensor::<ReferenceBackend, 1>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme);

        output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output.to_data().assert_approx_eq(&output_ref.to_data(), 3);
    }

    #[test]
    fn should_quantize_dequantize_per_block_symmetric() {
        // block_size > line_size
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Symmetric,
            QuantizationType::QInt8,
            BlockLayout::Flat(8),
        );
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [-1.8, -1.0, 0.0, 0.5, -0.8, 1.2, 0.25, 0.5],
                [-0.08, 0.12, 0.025, 0.05, 0.2, 0.3, 0.4, 0.5],
            ],
            &Default::default(),
        );
        let input_ref =
            Tensor::<ReferenceBackend, 2>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme);

        output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output.to_data().assert_approx_eq(&output_ref.to_data(), 3);
    }

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
    fn should_quantize_dequantize_per_block_affine() {
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Affine,
            QuantizationType::QInt8,
            BlockLayout::Flat(4),
        );
        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [-1.8, -1.0, 0.0, 0.5, -0.8, 1.2, 0.25, 0.5],
                [-0.08, 0.12, 0.025, 0.05, 0.2, 0.3, 0.4, 0.5],
            ],
            &Default::default(),
        );
        let input_ref =
            Tensor::<ReferenceBackend, 2>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme);

        let scales: [f32; 4] = [0.009019608, 0.007843138, 0.00078431366, 0.0019607844];
        let offsets: [i8; 4] = [71, -26, -25, -128];
        let expected = TensorData::quantized(
            vec![
                [-128i8, -40, 71, 126],
                [-128, 127, 6, 38],
                [-127, 127, 7, 39],
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

        println!("Ref check");
        assert_eq!(output_ref.to_data(), expected);
        println!("Output check");
        let output_data = output.to_data();
        if output_data != expected {
            println!("Values: {:?}", output_data.iter::<i8>().collect::<Vec<_>>());
            // Quantization parameters check
            let qparams = get_q_params(output_data);
            let expected = get_q_params(expected);
            assert_eq!(qparams.scale.len(), 4);
            if qparams.scale != expected.scale {
                println!(
                    "Scale not equal:\n{:?}\n{:?}",
                    qparams.scale, expected.scale
                );
            }
            assert_eq!(qparams.offset.as_ref().unwrap().len(), 4);
            if qparams.scale != expected.scale {
                println!(
                    "Offset not equal:\n{:?}\n{:?}",
                    qparams.offset, expected.offset
                );
            }
            panic!("Output != expected");
        }
        assert_eq!(output.to_data(), expected);
        output.to_data().assert_eq(&expected, true);
        output_ref.to_data().assert_eq(&expected, true);
        // output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output.to_data().assert_approx_eq(&output_ref.to_data(), 3);
    }
}
