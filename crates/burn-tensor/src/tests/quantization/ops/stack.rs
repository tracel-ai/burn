#[burn_tensor_testgen::testgen(q_stack)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_stack_ops_2d_dim0() {
        let device = Default::default();
        // Quantized [[1.0, 2.0, 3.0]]
        let data = TensorData::quantized(
            vec![-43i8, 42, 127],
            [1, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[4.0, 5.0, 6.0]]
        let data = TensorData::quantized(
            vec![42i8, 85, 127],
            [1, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.023529412, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_stack_ops_2d_dim1() {
        let device = Default::default();
        // Quantized [[1.0, 2.0, 3.0]]
        let data = TensorData::quantized(
            vec![-43i8, 42, 127],
            [1, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[4.0, 5.0, 6.0]]
        let data = TensorData::quantized(
            vec![42i8, 85, 127],
            [1, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.023529412, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 1);
        let expected = TensorData::from([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_stack_ops_3d() {
        let device = Default::default();
        // Quantized [[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]]
        let data = TensorData::quantized(
            vec![-43i8, 42, 127, 127, 42, -43],
            [2, 1, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor_1 = TestTensor::<3>::from_data(data, &device);
        // Quantized [[[4.0, 5.0, 6.0]], [[6.0, 5.0, 4.0]]]
        let data = TensorData::quantized(
            vec![42i8, 85, 127, 127, 85, 42],
            [2, 1, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.023529412, -128)),
        );
        let tensor_2 = TestTensor::<3>::from_data(data, &device);

        let output = Tensor::stack::<4>(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([
            [[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]],
            [[[4.0, 5.0, 6.0]], [[6.0, 5.0, 4.0]]],
        ]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_dimensions_are_not_the_same() {
        let device = Default::default();
        // Quantized [[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]
        let data = TensorData::quantized(
            vec![-43i8, 42, 127, 127, 42, -43],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[4.0, 5.0]]
        let data = TensorData::quantized(
            vec![76i8, 127],
            [1, 2],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output: Tensor<TestBackend, 3> = Tensor::stack(vec![tensor_1, tensor_2], 0);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_stack_exceeds_dimension() {
        let device = Default::default();
        // Quantized [[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]]
        let data = TensorData::quantized(
            vec![-43i8, 42, 127, 127, 42, -43],
            [1, 2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[4.0, 5.0]]
        let data = TensorData::quantized(
            vec![42i8, 85, 127],
            [1, 1, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.023529412, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output: Tensor<TestBackend, 4> = TestTensor::stack(vec![tensor_1, tensor_2], 3);
    }
}
