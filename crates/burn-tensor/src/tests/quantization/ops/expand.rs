#[burn_tensor_testgen::testgen(q_expand)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Shape, Tensor, TensorData};

    #[test]
    fn expand_2d() {
        // Quantized [1.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![42i8, 85, 127],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let output = tensor.expand([3, 3]);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq(
            &TensorData::from([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            1,
        );

        // Quantized [4.0, 7.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![73i8, 127, 36, 54],
            [4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let output = tensor.expand([2, 4]);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq(
            &TensorData::from([[4.0, 7.0, 2.0, 3.0], [4.0, 7.0, 2.0, 3.0]]),
            1,
        );
    }

    #[test]
    fn expand_3d() {
        // Quantized [[1.0, 2.0], [3.0, 4.0]]
        let data = TensorData::quantized(
            vec![32i8, 64, 95, 127],
            [2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.031496063)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());
        let output = tensor.expand([3, 2, 2]);
        let expected = TensorData::from([
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn expand_higher_dimensions() {
        // Quantized [[1.0, 2.0, 3.0, 4.0]]
        let data = TensorData::quantized(
            vec![32i8, 64, 95, 127],
            [1, 4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.031496063)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());
        let output = tensor.expand([2, 3, 4]);
        let expected = TensorData::from([
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
        ]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn broadcast_single() {
        // Quantized [1.0]
        let data = TensorData::quantized(
            vec![127i8],
            [1],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.007874016)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let output = tensor.expand([2, 3]);

        output
            .dequantize()
            .into_data()
            .assert_eq(&TensorData::from([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), false);
    }

    #[test]
    #[should_panic]
    fn should_fail_expand_incompatible_shapes() {
        // Quantized [1.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![42i8, 85, 127],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let _expanded_tensor = tensor.expand([2, 2]);
    }

    #[test]
    fn should_all_negative_one() {
        // Quantized [1.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![42i8, 85, 127],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let output = tensor.expand([2, -1]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[1., 2., 3.], [1., 2., 3.]]), 1);
    }

    #[test]
    #[should_panic]
    fn should_panic_negative_one_on_non_existing_dim() {
        // Quantized [1.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![42i8, 85, 127],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let _expanded_tensor = tensor.expand([-1, 3]);
    }
}
