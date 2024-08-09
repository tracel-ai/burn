#[burn_tensor_testgen::testgen(q_narrow)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Shape, Tensor, TensorData};

    #[test]
    fn test_narrow() {
        // Quantized [[1., 2., 3.], [7., 8., 9.], [13., 14., 15.]]
        let data = TensorData::quantized(
            vec![-111i8, -94, -77, -9, 8, 25, 93, 110, 127],
            [3, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        let output = tensor.clone().narrow(0, 0, 2);
        let expected = TensorData::from([[1., 2., 3.], [7., 8., 9.]]);

        assert_eq!(output.shape(), Shape::from([2, 3]));
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);

        let output = tensor.narrow(1, 1, 2);
        let expected = TensorData::from([[2., 3.], [8., 9.], [14., 15.]]);
        assert_eq!(output.shape(), Shape::from([3, 2]));
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }

    #[test]
    #[should_panic]
    fn test_narrow_invalid_dim() {
        // Quantized [[1., 2., 3.], [7., 8., 9.], [13., 14., 15.]]
        let data = TensorData::quantized(
            vec![-111i8, -94, -77, -9, 8, 25, 93, 110, 127],
            [3, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        let output = tensor.narrow(2, 0, 2);
    }

    #[test]
    #[should_panic]
    fn test_narrow_invalid_start() {
        // Quantized [[1., 2., 3.], [7., 8., 9.], [13., 14., 15.]]
        let data = TensorData::quantized(
            vec![-111i8, -94, -77, -9, 8, 25, 93, 110, 127],
            [3, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        let output = tensor.narrow(0, 3, 2);
    }

    #[test]
    #[should_panic]
    fn test_narrow_invalid_zero_length() {
        // Quantized [[1., 2., 3.], [7., 8., 9.], [13., 14., 15.]]
        let data = TensorData::quantized(
            vec![-111i8, -94, -77, -9, 8, 25, 93, 110, 127],
            [3, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        let output = tensor.narrow(0, 1, 0);
    }

    #[test]
    #[should_panic]
    fn test_narrow_invalid_length() {
        // Quantized [[1., 2., 3.], [7., 8., 9.], [13., 14., 15.]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [3, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data.clone(), &Default::default());

        let output = tensor.narrow(0, 0, 4);
    }
}
