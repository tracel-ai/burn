#[burn_tensor_testgen::testgen(q_select)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_select_1d() {
        let device = Default::default();
        // Quantized [0.0, 1.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![-128i8, -43, 42, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data, &device);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

        let output = tensor.select(0, indices);
        let expected = TensorData::from([1.0, 1.0, 0.0, 1.0, 2.0]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_2d_dim0_same_num_dim() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::from_data(([1, 0]), &device);

        let output = tensor.select(0, indices);
        let expected = TensorData::from([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_2d_dim0_more_num_dim() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::from_data([1, 0, 1, 1], &device);

        let output = tensor.select(0, indices);
        let expected = TensorData::from([
            [3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0],
        ]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_2d_dim1() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

        let output = tensor.select(1, indices);
        let expected = TensorData::from([[1.0, 1.0, 0.0, 1.0, 2.0], [4.0, 4.0, 3.0, 4.0, 5.0]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_assign_1d() {
        let device = Default::default();
        // Quantized [0.0, 1.0, 2.0]
        let data = TensorData::quantized(
            vec![-128i8, -1, 127],
            [3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.007843138, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data, &device);
        // Quantized [5.0, 4.0, 3.0, 2.0, 1.0]
        let data = TensorData::quantized(
            vec![127i8, 76, 25, -26, -77],
            [5],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let values = TestTensor::<1>::from_data(data, &device);
        let indices = TestTensorInt::from_data(TensorData::from([1, 1, 0, 1, 2]), &device);

        let output = tensor.select_assign(0, indices, values);
        let expected = TensorData::from([3.0, 12.0, 3.0]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_select_assign_2d_dim0() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let values = tensor.clone();
        let indices = TestTensorInt::from_data(TensorData::from([1, 0]), &device);

        let output = tensor.select_assign(0, indices, values);
        let expected = TensorData::from([[3.0, 5.0, 7.0], [3.0, 5.0, 7.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_select_assign_2d_dim1() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let values = tensor.clone();
        let indices = TestTensorInt::from_data(TensorData::from([1, 0, 2]), &device);

        let output = tensor.select_assign(1, indices, values);
        let expected = TensorData::from([[1.0, 1.0, 4.0], [7.0, 7.0, 10.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    #[should_panic]
    fn should_select_panic_invalid_dimension() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &device);

        tensor.select(10, indices);
    }
}
