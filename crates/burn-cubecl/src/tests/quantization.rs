#[burn_tensor_testgen::testgen(quantization)]
mod tests {
    use super::*;
    use burn_tensor::{
        Int, Tensor,
        quantization::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue},
    };
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_quantize_dequantize_symmetric_arange() {
        let scheme = QuantScheme::default();
        let input = Tensor::<TestBackend, 1, Int>::arange(0..128, &Default::default()).float();
        let input_ref =
            Tensor::<ReferenceBackend, 1>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme);

        output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output
            .into_data()
            .assert_approx_eq::<FT>(&output_ref.to_data(), Tolerance::default());
    }

    #[test]
    fn should_quantize_dequantize_symmetric_per_block() {
        let scheme = QuantScheme::default().with_level(QuantLevel::Block(8));

        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [
                    -1.8, -1.0, 0.0, 0.5, -1.8, -1.0, 0.0, 0.5, 0.01, 0.025, 0.03, 0.04, 0.01,
                    0.025, 0.03, 0.04,
                ],
                [
                    1.8, 1.0, 0.0, -0.5, 1.8, 1.0, 0.0, -0.5, -0.01, -0.025, -0.03, -0.04, -0.01,
                    -0.025, -0.03, -0.04,
                ],
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

        output
            .into_data()
            .assert_approx_eq::<FT>(&output_ref.to_data(), Tolerance::default());
    }
}
