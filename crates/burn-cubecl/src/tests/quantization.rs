#[burn_tensor_testgen::testgen(quantization)]
mod tests {
    use super::*;
    use burn_tensor::{
        quantization::{QuantizationMode, QuantizationScheme, QuantizationType},
        Tensor,
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
}
