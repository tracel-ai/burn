#[burn_tensor_testgen::testgen(q_repeat_dim)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_repeat_ops() {
        // Quantized [[0.0, 1.0, 2.0, 3.0]]
        let data = TensorData::quantized(
            vec![-128i8, -43, 42, 127],
            [1, 4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.repeat_dim(0, 4);
        let expected = TensorData::from([
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        ]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_repeat_on_dims_larger_than_1() {
        // Quantized [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
        let data = TensorData::quantized(
            vec![
                -128i8, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
            ],
            [4, 2, 2],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let output = tensor.repeat_dim(2, 2);
        let expected = TensorData::from([
            [[0., 1., 0., 1.], [2., 3., 2., 3.]],
            [[4., 5., 4., 5.], [6., 7., 6., 7.]],
            [[8., 9., 8., 9.], [10., 11., 10., 11.]],
            [[12., 13., 12., 13.], [14., 15., 14., 15.]],
        ]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }
}
