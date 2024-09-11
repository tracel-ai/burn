#[burn_tensor_testgen::testgen(q_slice)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Int, Tensor, TensorData};

    #[test]
    fn should_support_full_sliceing_1d() {
        // Quantized [0.0, 1.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![-128i8, -43, 42, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([0..4]);

        output.into_data().assert_eq(&data, false);
    }

    #[test]
    fn should_support_partial_sliceing_1d() {
        // Quantized [0.0, 1.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![-128i8, -43, 42, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.slice([1..3]);
        let expected = TensorData::from([1.0, 2.0]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_full_sliceing_2d() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        let output = tensor.clone().slice([0..2]);
        output.into_data().assert_eq(&data, true);

        let output = tensor.slice([0..2, 0..3]);
        output.into_data().assert_eq(&data, true);
    }

    #[test]
    fn should_support_partial_sliceing_2d() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.slice([0..2, 0..2]);
        let expected = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_partial_sliceing_3d() {
        // Quantized [[[0., 1., 2., 3.], [4., 5., 6., 7.]], [[8., 9., 10., 11.], [12., 13., 14., 15.]]]
        let data = TensorData::quantized(
            vec![
                -128i8, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
            ],
            [2, 2, 4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let output = tensor.slice([1..2, 1..2, 0..2]);
        let expected = TensorData::from([[[12.0, 13.0]]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_partial_sliceing_3d_non_contiguous() {
        // Quantized [[[0., 1., 2., 3.], [4., 5., 6., 7.]], [[8., 9., 10., 11.], [12., 13., 14., 15.]]]
        let data = TensorData::quantized(
            vec![
                -128i8, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
            ],
            [2, 2, 4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let output = tensor.transpose().slice([1..2, 1..2, 0..2]);
        let expected = TensorData::from([[[9.0, 13.0]]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_slice_assign_1d() {
        let device = Default::default();
        // Quantized [0.0, 1.0, 2.0]
        let data = TensorData::quantized(
            vec![-128i8, -1, 127],
            [3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.007843138, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data, &device);
        // Quantized [10.0, 5.0]
        let data = TensorData::quantized(
            vec![127i8, -1],
            [2],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.039215688, -128)),
        );
        let tensor_assigned = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.slice_assign([0..2], tensor_assigned);
        let expected = TensorData::from([10.0, 5.0, 2.0]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_slice_assign_2d() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        // Quantized [[10.0, 5.0]]
        let data = TensorData::quantized(
            vec![127i8, -1],
            [1, 2],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.039215688, -128)),
        );
        let tensor_assigned = Tensor::<TestBackend, 2>::from_data(data, &device);

        let output = tensor.slice_assign([1..2, 0..2], tensor_assigned);
        let expected = TensorData::from([[0.0, 1.0, 2.0], [10.0, 5.0, 5.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn slice_should_not_corrupt_potentially_inplace_operations() {
        // Quantized [1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![-77i8, -26, 25, 76, 127],
            [5],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let tensor = tensor.clone().slice([0..3]) + tensor.clone().slice([2..5]);

        let expected = TensorData::from([4., 6., 8.]);

        // Precision 1 to approximate de/quantization errors
        tensor
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn slice_assign_should_not_corrupt_potentially_inplace_operations() {
        let device = Default::default();
        // Quantized [1.0, 2.0, 3.0, 4.0, 5.0]
        let data = TensorData::quantized(
            vec![-77i8, -26, 25, 76, 127],
            [5],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data, &device);
        // Quantized [10., 20., 30.]
        let data = TensorData::quantized(
            vec![-43i8, 42, 127],
            [3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.11764706, -128)),
        );
        let values = TestTensor::<1>::from_data(data, &device);
        let tensor_1 = tensor.clone().slice_assign([0..3], values);
        let tensor_2 = tensor + 2;

        let expected = TensorData::from([10., 20., 30., 4., 5.]);

        // Precision 1 to approximate de/quantization errors
        tensor_1
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);

        let expected = TensorData::from([3., 4., 5., 6., 7.]);

        // Precision 1 to approximate de/quantization errors
        tensor_2
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn clamp_when_slice_exceeds_dimension() {
        // Quantized [0.0, 1.0, 2.0]
        let data = TensorData::quantized(
            vec![-128i8, -1, 127],
            [3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.007843138, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([0..4]);
        output.into_data().assert_eq(&data, true);
    }

    #[test]
    fn negative_dimensions() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        // Clamping to the tensor dimensions
        let output = tensor.clone().slice([(0, 4), (0, 4)]);
        output.into_data().assert_eq(&data, true);

        // Negative dimensions
        let output = tensor.clone().slice([(0, 1), (0, 1)]);
        let data = TensorData::from([[0.0f32]]);
        output.dequantize().into_data().assert_eq(&data, false);

        let output = tensor.slice([(0, -1), (0, -2)]);
        output.dequantize().into_data().assert_eq(&data, false);
    }

    #[test]
    fn missing_dimensions() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        // Clamping to the tensor dimensions
        let output = tensor.clone().slice([Some((0, 4)), Some((0, 4))]);
        output.into_data().assert_eq(&data, true);

        // Negative dimensions
        let data = TensorData::from([[0.0f32]]);
        let output = tensor.clone().slice([Some((0, -1)), Some((0, -2))]);
        output.dequantize().into_data().assert_eq(&data, false);

        // Missing dimensions
        let output = tensor.clone().slice([Some((0, 1)), None]);
        let data = TensorData::from([[0.0f32, 1.0, 2.0]]);
        output.dequantize().into_data().assert_eq(&data, false);

        let output = tensor.clone().slice([None, Some((0, 2))]);
        let data = TensorData::from([[0.0f32, 1.0], [3.0, 4.0]]);
        output.dequantize().into_data().assert_eq(&data, false);

        let output = tensor.clone().slice([None, None]);
        let data = TensorData::from([[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        output.dequantize().into_data().assert_eq(&data, false);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_with_too_many_dimensions() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([0..1, 0..1]);

        output.into_data().assert_eq(&data, false);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_is_desc() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone(), &Default::default());

        #[allow(clippy::reversed_empty_ranges)]
        let output = tensor.slice([2..1]);

        output.into_data().assert_eq(&data, false);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_slice_is_equal() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data.clone(), &Default::default());

        let output = tensor.slice([1..1]);

        output.into_data().assert_eq(&data, false);
    }
}
