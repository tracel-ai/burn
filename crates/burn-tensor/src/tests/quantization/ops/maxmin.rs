#[burn_tensor_testgen::testgen(q_maxmin)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_max_dim_2d() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.max_dim(1);
        let expected = TensorData::from([[2.], [5.]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_max_dim_with_indices_2d_with_dim_0th() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let (output, index) = tensor.max_dim_with_indices(0);

        let output_expected = TensorData::from([[3., 4., 5.]]);
        let index_expected = TensorData::from([[1, 1, 1]]);

        output
            .dequantize()
            .into_data()
            .assert_eq(&output_expected, false);
        index.into_data().assert_eq(&index_expected, false);
    }

    #[test]
    fn test_max_dim_with_indices_2d() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let (output, index) = tensor.max_dim_with_indices(1);

        let output_expected = TensorData::from([[2.], [5.]]);
        let index_expected = TensorData::from([[2], [2]]);

        output
            .dequantize()
            .into_data()
            .assert_eq(&output_expected, false);
        index.into_data().assert_eq(&index_expected, false);
    }

    #[test]
    fn test_min_dim_2d() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.min_dim(1);

        let expected = TensorData::from([[0.], [3.]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_min_dim_with_indices_2d() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let (output, index) = tensor.min_dim_with_indices(1);

        let output_expected = TensorData::from([[0.], [3.]]);
        let index_expected = TensorData::from([[0], [0]]);

        output
            .dequantize()
            .into_data()
            .assert_eq(&output_expected, false);
        index.into_data().assert_eq(&index_expected, false);
    }

    #[test]
    fn test_min_dim_2d_with_0th_dim() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.min_dim(0);
        let expected = TensorData::from([[0., 1., 2.]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_max_dim_2d_with_0th_dim() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.max_dim(0);
        let expected = TensorData::from([[3., 4., 5.]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_min_dim_with_indices_2d_with_0th_dim() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let (output, index) = tensor.min_dim_with_indices(0);

        let output_expected = TensorData::from([[0., 1., 2.]]);
        let index_expected = TensorData::from([[0, 0, 0]]);

        output
            .dequantize()
            .into_data()
            .assert_eq(&output_expected, false);
        index.into_data().assert_eq(&index_expected, false);
    }

    #[test]
    fn test_maximum_pair() {
        // Quantized [1.0, 2.0, 3.0, 4.0] (with range [0., 5.])
        let data = TensorData::quantized(
            vec![-77i8, -26, 25, 76],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let a = TestTensor::<1>::from_data(data, &Default::default());
        // Quantized [2.0, 1.0, 4.0, 5.0] (with range [0., 5.])
        let data = TensorData::quantized(
            vec![-26i8, -77, 76, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let b = TestTensor::<1>::from_data(data, &Default::default());

        let output = a.max_pair(b);
        let expected = TensorData::from([2.0, 2.0, 4.0, 5.0]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn test_minimum_pair() {
        // Quantized [1.0, 2.0, 3.0, 4.0] (with range [0., 5.])
        let data = TensorData::quantized(
            vec![-77i8, -26, 25, 76],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let a = TestTensor::<1>::from_data(data, &Default::default());
        // Quantized [2.0, 1.0, 4.0, 5.0] (with range [0., 5.])
        let data = TensorData::quantized(
            vec![-26i8, -77, 76, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let b = TestTensor::<1>::from_data(data, &Default::default());

        let output = a.min_pair(b);
        let expected = TensorData::from([1.0, 1.0, 3.0, 4.0]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
