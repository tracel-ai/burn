#[burn_tensor_testgen::testgen(q_permute)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Device, Tensor, TensorData};

    #[test]
    fn permute_float() {
        let device = Default::default();
        // Quantized [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
        let data = TensorData::quantized(
            vec![
                -128i8, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
            ],
            [2, 2, 4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<3>::from_data(data.clone(), &device);

        let permuted = tensor.clone().permute([2, 1, 0]);

        let expected = TensorData::from([
            [[0., 8.], [4., 12.]],
            [[1., 9.], [5., 13.]],
            [[2., 10.], [6., 14.]],
            [[3., 11.], [7., 15.]],
        ]);

        permuted
            .dequantize()
            .into_data()
            .assert_eq(&expected, false);

        // Test with negative axis
        let permuted = tensor.clone().permute([-1, 1, 0]);
        permuted
            .dequantize()
            .into_data()
            .assert_eq(&expected, false);

        // Test with the same axis
        let permuted = tensor.clone().permute([0, 1, 2]);
        permuted.into_data().assert_eq(&tensor.into_data(), true);
    }

    #[test]
    #[should_panic]
    fn edge_repeated_axes() {
        let device = Default::default();
        // Quantized [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
        let data = TensorData::quantized(
            vec![
                -128i8, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
            ],
            [2, 2, 4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<3>::from_data(data.clone(), &device);

        // Test with a repeated axis
        let _ = tensor.clone().permute([0, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn edge_out_of_bound_axis() {
        let device = Default::default();
        // Quantized [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
        let data = TensorData::quantized(
            vec![
                -128i8, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
            ],
            [2, 2, 4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor = TestTensor::<3>::from_data(data.clone(), &device);

        // Test with an invalid axis
        let _ = tensor.clone().permute([3, 0, 1]);
    }
}
