#[burn_tensor_testgen::testgen(quantization)]
mod tests {
    use super::*;
    use burn_tensor::{
        Int, Tensor,
        backend::Backend,
        quantization::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue},
    };
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    fn should_quantize_dequantize_symmetric_arange(store: QuantStore) {
        let scheme = QuantScheme::default().with_store(store);
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

    fn should_quantize_dequantize_symmetric_per_block(store: QuantStore) {
        let scheme = QuantScheme::default()
            .with_level(QuantLevel::Block(8))
            .with_store(store);

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

    fn supports_native() -> bool {
        let name = <TestBackend as Backend>::name(&Default::default());
        // TODO: Proper checks for i8 support.
        name.contains("cuda")
            || name.contains("rocm")
            || name.contains("hip")
            || name.contains("vulkan")
            || name.contains("spirv")
            || name.contains("metal")
            || name.contains("msl")
    }

    #[test]
    fn should_quantize_dequantize_symmetric_arange_packed() {
        should_quantize_dequantize_symmetric_arange(QuantStore::U32)
    }

    #[test]
    fn should_quantize_dequantize_symmetric_per_block_packed() {
        should_quantize_dequantize_symmetric_per_block(QuantStore::U32)
    }

    #[test]
    fn should_quantize_dequantize_symmetric_arange_native() {
        if supports_native() {
            should_quantize_dequantize_symmetric_arange(QuantStore::Native)
        }
    }

    #[test]
    fn should_quantize_dequantize_symmetric_per_block_native() {
        if supports_native() {
            should_quantize_dequantize_symmetric_per_block(QuantStore::Native)
        }
    }

    #[test]
    #[should_panic = "Can't store in u32"]
    fn should_panic_when_shape_cannot_store_quants() {
        let device = Default::default();
        let scheme = QuantScheme::default();

        let tensor_1 =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 6.35], [2.0, 3.0], [1.0, 3.0]], &device)
                .quantize_dynamic(&scheme);
    }
}
