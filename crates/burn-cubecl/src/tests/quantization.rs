#[burn_tensor_testgen::testgen(quantization)]
mod tests {
    use super::*;
    use burn_tensor::{
        Int, Shape, Tensor,
        backend::Backend,
        quantization::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue},
    };
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    fn should_quantize_dequantize_symmetric_arange<S: Into<Shape>>(
        value: QuantValue,
        store: QuantStore,
        shape: S,
    ) {
        let shape = shape.into();
        assert_eq!(shape.rank(), 2); // 2D tests

        let scheme = QuantScheme::default().with_value(value).with_store(store);
        let scheme_ref = scheme.clone().with_store(QuantStore::Native);

        let input: Tensor<TestBackend, 2> =
            Tensor::arange(0..shape.num_elements() as i64, &Default::default())
                .float()
                .reshape(shape);
        let input_ref =
            Tensor::<ReferenceBackend, 2>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme_ref);

        output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output
            .into_data()
            .assert_approx_eq::<FT>(&output_ref.to_data(), Tolerance::default());
    }

    fn should_quantize_dequantize_symmetric_per_block_arange<S: Into<Shape>>(
        value: QuantValue,
        block_size: usize,
        store: QuantStore,
        shape: S,
    ) {
        let scheme = QuantScheme::default()
            .with_value(value)
            .with_level(QuantLevel::block([block_size as u8]))
            .with_store(store);
        let scheme_ref = scheme.clone().with_store(QuantStore::Native);

        let shape = shape.into();
        let input: Tensor<TestBackend, 2> =
            Tensor::arange(0..shape.num_elements() as i64, &Default::default())
                .float()
                .reshape(shape);
        let input_ref =
            Tensor::<ReferenceBackend, 2>::from_data(input.to_data(), &Default::default());

        let output = input.quantize_dynamic(&scheme);
        let output_ref = input_ref.quantize_dynamic(&scheme_ref);

        output.to_data().assert_eq(&output_ref.to_data(), false);

        let output = output.dequantize();
        let output_ref = output_ref.dequantize();

        output
            .into_data()
            .assert_approx_eq::<FT>(&output_ref.to_data(), Tolerance::default());
    }

    fn should_quantize_dequantize_symmetric_per_block(
        value: QuantValue,
        block_size: usize,
        store: QuantStore,
    ) {
        let scheme = QuantScheme::default()
            .with_value(value)
            .with_level(QuantLevel::block([block_size as u8]))
            .with_store(store);
        let scheme_ref = scheme.clone().with_store(QuantStore::Native);

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
        let output_ref = input_ref.quantize_dynamic(&scheme_ref);

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
    fn should_quantize_dequantize_symmetric_arange_q8s_packed() {
        should_quantize_dequantize_symmetric_arange(QuantValue::Q8S, QuantStore::U32, [8, 16])
    }

    #[test]
    fn should_quantize_dequantize_symmetric_arange_q8f_packed() {
        should_quantize_dequantize_symmetric_arange(QuantValue::Q8F, QuantStore::U32, [8, 16])
    }

    #[test]
    fn should_quantize_dequantize_symmetric_arange_q4s_packed() {
        should_quantize_dequantize_symmetric_arange(QuantValue::Q4S, QuantStore::U32, [8, 16])
    }

    #[test]
    fn should_quantize_dequantize_symmetric_arange_q4f_packed() {
        should_quantize_dequantize_symmetric_arange(QuantValue::Q4F, QuantStore::U32, [8, 16])
    }

    #[test]
    fn should_quantize_dequantize_symmetric_arange_q2s_packed() {
        should_quantize_dequantize_symmetric_arange(QuantValue::Q2S, QuantStore::U32, [8, 16])
    }

    #[test]
    fn should_quantize_dequantize_symmetric_arange_q2f_packed() {
        should_quantize_dequantize_symmetric_arange(QuantValue::Q2F, QuantStore::U32, [8, 16])
    }

    #[test]
    fn should_quantize_dequantize_symmetric_per_block_q8s_packed() {
        should_quantize_dequantize_symmetric_per_block(QuantValue::Q8S, 8, QuantStore::U32)
    }

    #[test]
    fn should_quantize_dequantize_symmetric_per_block_q4s_packed() {
        should_quantize_dequantize_symmetric_per_block(QuantValue::Q4S, 8, QuantStore::U32)
    }

    #[test]
    #[should_panic = "Block size must be divisible by 16"]
    fn should_panic_when_block_size_cannot_store_num_quants() {
        // num_quants in u32 = 32 bits / 2 bits = 16
        should_quantize_dequantize_symmetric_per_block(QuantValue::Q2S, 8, QuantStore::U32)
    }

    #[test]
    fn should_quantize_dequantize_symmetric_per_block_q2s_packed() {
        should_quantize_dequantize_symmetric_per_block(QuantValue::Q2S, 16, QuantStore::U32)
    }

    #[test]
    fn should_quantize_dequantize_symmetric_arange_q8s_native() {
        if supports_native() {
            should_quantize_dequantize_symmetric_arange(
                QuantValue::Q8S,
                QuantStore::Native,
                [32, 32],
            )
        }
    }

    #[test]
    fn should_quantize_dequantize_symmetric_per_block_q8s_native() {
        if supports_native() {
            should_quantize_dequantize_symmetric_per_block(QuantValue::Q8S, 8, QuantStore::Native)
        }
    }

    #[test]
    fn should_quantize_dequantize_symmetric_per_block_arange_q8s_packed() {
        should_quantize_dequantize_symmetric_per_block_arange(
            QuantValue::Q8S,
            32,
            QuantStore::U32,
            [32, 32],
        )
    }

    #[test]
    fn should_quantize_dequantize_symmetric_per_block_arange_q8s_native() {
        if supports_native() {
            should_quantize_dequantize_symmetric_per_block_arange(
                QuantValue::Q8S,
                32,
                QuantStore::Native,
                [32, 32],
            )
        }
    }

    #[test]
    fn should_quantize_dequantize_symmetric_arange_128x256_q8s_native() {
        if supports_native() {
            should_quantize_dequantize_symmetric_per_block_arange(
                QuantValue::Q8S,
                32,
                QuantStore::Native,
                [128, 256],
            )
        }
    }
    #[test]
    fn should_quantize_dequantize_symmetric_arange_128x256_q8s_packed() {
        should_quantize_dequantize_symmetric_per_block_arange(
            QuantValue::Q8S,
            32,
            QuantStore::U32,
            [128, 256],
        )
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
