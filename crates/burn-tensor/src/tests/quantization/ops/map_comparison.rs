#[burn_tensor_testgen::testgen(q_map_comparison)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    // NOTE: we use affine quantization to reduce quantization errors since equality tests are precise
    #[test]
    fn test_equal() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[0.0, 1.0, 1.0], [3.0, 5.0, 4.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -77, 25, 127, 76],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let data_actual_cloned = tensor_1.clone().equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.equal(tensor_2);

        let data_expected = TensorData::from([[true, true, false], [true, false, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_not_equal() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[0.0, 1.0, 1.0], [3.0, 5.0, 4.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -77, 25, 127, 76],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let data_actual_cloned = tensor_1.clone().not_equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.not_equal(tensor_2);

        let data_expected = TensorData::from([[false, false, true], [false, true, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_equal_elem() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 2.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, -26, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default());

        let data_actual_cloned = tensor_1.clone().equal_elem(2);
        let data_actual_inplace = tensor_1.equal_elem(2);

        let data_expected = TensorData::from([[false, false, true], [false, true, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_not_equal_elem() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 2.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, -26, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default());

        let data_actual_cloned = tensor_1.clone().not_equal_elem(2);
        let data_actual_inplace = tensor_1.not_equal_elem(2);

        let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_greater_elem() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default());

        let data_actual_cloned = tensor_1.clone().greater_elem(4);
        let data_actual_inplace = tensor_1.greater_elem(4);

        let data_expected = TensorData::from([[false, false, false], [false, false, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_greater_equal_elem() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default());

        let data_actual_cloned = tensor_1.clone().greater_equal_elem(4.0);
        let data_actual_inplace = tensor_1.greater_equal_elem(4.0);

        let data_expected = TensorData::from([[false, false, false], [false, true, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_greater() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[0.0, 1.0, 1.0], [3.0, 5.0, 4.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -77, 25, 127, 76],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let data_actual_cloned = tensor_1.clone().greater(tensor_2.clone());
        let data_actual_inplace = tensor_1.greater(tensor_2);

        let data_expected = TensorData::from([[false, false, true], [false, false, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_greater_equal() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 1.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -77, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[0.0, 1.0, 2.0], [3.0, 5.0, 4.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 127, 76],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let data_actual_cloned = tensor_1.clone().greater_equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.greater_equal(tensor_2);

        let data_expected = TensorData::from([[true, true, false], [true, false, true]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_lower_elem() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default());

        let data_actual_cloned = tensor_1.clone().lower_elem(4.0);
        let data_actual_inplace = tensor_1.lower_elem(4.0);

        let data_expected = TensorData::from([[true, true, true], [true, false, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_lower_equal_elem() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default());

        let data_actual_cloned = tensor_1.clone().lower_equal_elem(4.0);
        let data_actual_inplace = tensor_1.lower_equal_elem(4.0);

        let data_expected = TensorData::from([[true, true, true], [true, true, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_lower() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 1.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -77, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[0.0, 1.0, 2.0], [3.0, 5.0, 4.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 127, 76],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let data_actual_cloned = tensor_1.clone().lower(tensor_2.clone());
        let data_actual_inplace = tensor_1.lower(tensor_2);

        let data_expected = TensorData::from([[false, false, true], [false, true, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }

    #[test]
    fn test_lower_equal() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[0.0, 1.0, 1.0], [3.0, 5.0, 4.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -77, 25, 127, 76],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let data_actual_cloned = tensor_1.clone().lower_equal(tensor_2.clone());
        let data_actual_inplace = tensor_1.lower_equal(tensor_2);

        let data_expected = TensorData::from([[true, true, false], [true, true, false]]);
        assert_eq!(data_expected, data_actual_cloned.into_data());
        assert_eq!(data_expected, data_actual_inplace.into_data());
    }
}
