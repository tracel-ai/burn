#[burn_tensor_testgen::testgen(q_topk)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Shape, Tensor, TensorData};

    #[test]
    fn test_topk_1d() {
        // Quantized [1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![-77i8, -26, 25, 76, 127],
            [5],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let values = tensor.topk(3, /*dim*/ 0);
        let expected = TensorData::from([5., 4., 3.]);

        values
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }

    #[test]
    fn test_topk() {
        // Quantized [[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 7.]]]
        let data = TensorData::quantized(
            vec![-100i8, -15, 70, -71, 14, 42, -43, -128, 127, 99, -71, 70],
            [2, 2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.03529412, -128)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let values = tensor.topk(2, /*dim*/ 2);
        let expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);

        // Precision 1 to approximate de/quantization errors
        values
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn test_topk_with_indices() {
        // 1D
        // Quantized [1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![-77i8, -26, 25, 76, 127],
            [5],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let (values, indices) = tensor.topk_with_indices(3, /*dim*/ 0);

        let values_expected = TensorData::from([5., 4., 3.]);
        values
            .dequantize()
            .into_data()
            .assert_eq(&values_expected, false);

        let indices_expected = TensorData::from([4, 3, 2]);
        indices.into_data().assert_eq(&indices_expected, false);

        // 3D
        // Quantized [[[1., 4., 7.], [2., 5., 6.]], [[3., 0., 9.], [8., 2., 7.]]]
        let data = TensorData::quantized(
            vec![-100i8, -15, 70, -71, 14, 42, -43, -128, 127, 99, -71, 70],
            [2, 2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.03529412, -128)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let (values, indices) = tensor.topk_with_indices(2, /*dim*/ 2);

        let values_expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);

        // Precision 1 to approximate de/quantization errors
        values
            .dequantize()
            .into_data()
            .assert_approx_eq(&values_expected, 1);

        let indices_expected = TensorData::from([[[2, 1], [2, 1]], [[2, 0], [0, 2]]]);

        indices.into_data().assert_eq(&indices_expected, false);
    }
}
