#[burn_tensor_testgen::testgen(quantization)]
mod tests {
    use super::*;
    use burn_tensor::{
        Tensor,
        quantization::{QuantizationMode, QuantizationScheme, QuantizationType},
    };
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_quantize_dequantize_symmetric_single() {
        let scheme = QuantizationScheme::default();
        let input = Tensor::<TestBackend, 1>::from_floats([-1.8], &Default::default());
        let input_ref =
            Tensor::<ReferenceBackend, 1>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme);

        output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output
            .to_data()
            .assert_approx_eq::<FT>(&output_ref.to_data(), Tolerance::rel_abs(1e-2, 1e-2));
    }

    #[test]
    fn should_quantize_dequantize_symmetric_multiple() {
        let scheme = QuantizationScheme::default();
        let input =
            Tensor::<TestBackend, 1>::from_floats([-1.8, -1.0, 0.0, 0.5, 0.0], &Default::default());
        let input_ref =
            Tensor::<ReferenceBackend, 1>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme);

        output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output
            .to_data()
            .assert_approx_eq::<FT>(&output_ref.to_data(), Tolerance::rel_abs(1e-2, 1e-2));
    }
}
