#[burn_tensor_testgen::testgen(q_matmul)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{
        AffineQuantization, QuantizationStrategy, SymmetricQuantization,
    };
    use burn_tensor::{Int, Tensor, TensorData};

    // NOTE: we set higher tolerance (0.3) due to larger de/quantization errors accumulation
    #[test]
    fn test_matmul_d2() {
        let device = Default::default();
        // Quantized [[1.0, 7.0], [2.0, 3.0], [1.0, 5.0]]
        let data = TensorData::quantized(
            vec![18i8, 127, 36, 54, 18, 91],
            [3, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]]
        let data = TensorData::quantized(
            vec![73i8, 127, 91, 36, 54, 91],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected =
            TensorData::from([[18.0, 28.0, 40.0], [14.0, 23.0, 25.0], [14.0, 22.0, 30.0]]);

        tensor_3
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    fn test_matmul_d3() {
        let device = Default::default();
        // Quantized [[[1.0, 7.0], [2.0, 3.0]]]
        let data = TensorData::quantized(
            vec![18i8, 127, 36, 54],
            [1, 2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor_1 = TestTensor::<3>::from_data(data, &device);
        // Quantized [[[4.0, 7.0], [2.0, 3.0]]]
        let data = TensorData::quantized(
            vec![73i8, 127, 36, 54],
            [1, 2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor_2 = TestTensor::<3>::from_data(data, &device);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected = TensorData::from([[[18.0, 28.0], [14.0, 23.0]]]);

        tensor_3
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    fn test_matmul_broadcast_1() {
        let device = Default::default();
        // Quantized [[[1.0, 7.0], [2.0, 3.0]]]
        let data = TensorData::quantized(
            vec![18i8, 127, 36, 54],
            [1, 2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor_1 = TestTensor::<3>::from_data(data, &device);
        // Quantized [[[4.0, 7.0], [2.0, 3.0]], [[2.0, 5.0], [6.0, 3.0]]]
        let data = TensorData::quantized(
            vec![73i8, 127, 36, 54, 36, 91, 109, 54],
            [2, 2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor_2 = TestTensor::<3>::from_data(data, &device);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected =
            TensorData::from([[[18.0, 28.0], [14.0, 23.0]], [[44.0, 26.0], [22.0, 19.0]]]);

        tensor_3
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    fn test_matmul_broadcast_4d() {
        let device = Default::default();
        // Quantized [[[[1.0, 7.0], [2.0, 3.0]]], [[[2.0, 5.0], [6.0, 3.0]]]]
        let data = TensorData::quantized(
            vec![18i8, 127, 36, 54, 36, 91, 109, 54],
            [2, 1, 2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor_1 = TestTensor::<4>::from_data(data, &device);
        // Quantized [[[[9.0, 8.0], [1.0, 4.0]], [[2.0, 7.0], [3.0, 5.0]]]]
        let data = TensorData::quantized(
            vec![127i8, 113, 14, 56, 28, 99, 42, 71],
            [1, 2, 2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.070866145)),
        );
        let tensor_2 = TestTensor::<4>::from_data(data, &device);

        // [2, 1, 2, 2] @ [1, 2, 2, 2] -> [2, 2, 2, 2]
        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected = TensorData::from([
            [[[16.0, 36.0], [21.0, 28.0]], [[23.0, 42.0], [13.0, 29.0]]],
            [[[23.0, 36.0], [57.0, 60.0]], [[19.0, 39.0], [21.0, 57.0]]],
        ]);

        tensor_3
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    fn test_matmul_simple_1() {
        let device = Default::default();
        // NOTE: we use affine quantization to lower de/quantization errors
        // Quantized [[5.0, 14.0], [14.0, 25.0]]
        let data = TensorData::quantized(
            vec![-77i8, 15, 15, 127],
            [2, 2],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.09803922, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]
        let data = TensorData::quantized(
            vec![25i8, 76, 127, -128, -77, -26],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected = TensorData::from([[15.0, 34.0, 53.0], [42.0, 81.0, 120.0]]);

        tensor_3
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    fn test_matmul_4_3() {
        let device = Default::default();
        // NOTE: we use affine quantization to lower de/quantization errors
        // Quantized [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]]
        let data = TensorData::quantized(
            vec![-128i8, -105, -82, -58, -35, -12, 11, 34, 57, 81, 104, 127],
            [3, 4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.043137256, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[0., 1., 2.], [4., 5., 6.], [8., 9., 10.], [13., 14., 15.]]
        let data = TensorData::quantized(
            vec![-128i8, -111, -94, -60, -43, -26, 8, 25, 42, 93, 110, 127],
            [4, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected = TensorData::from([[59., 65., 71.], [159., 181., 203.], [259., 297., 335.]]);

        tensor_3
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    fn test_matmul_trivial() {
        let device = Default::default();
        // NOTE: we use affine quantization to lower de/quantization errors
        // Quantized [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.], [12., 13., 14., 15.]]
        let data = TensorData::quantized(
            vec![
                -128i8, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
            ],
            [4, 4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);

        let tensor_3 = tensor_1.clone().matmul(tensor_1);

        tensor_3.dequantize().into_data().assert_approx_eq(
            &TensorData::from([
                [56., 62., 68., 74.],
                [152., 174., 196., 218.],
                [248., 286., 324., 362.],
                [344., 398., 452., 506.],
            ]),
            3,
        );
    }

    #[test]
    fn test_matmul_trivial_transposed() {
        let device = Default::default();
        // NOTE: we use affine quantization to lower de/quantization errors
        // Quantized [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.], [12., 13., 14., 15.]]
        let data = TensorData::quantized(
            vec![
                -128i8, -111, -94, -77, -60, -43, -26, -9, 8, 25, 42, 59, 76, 93, 110, 127,
            ],
            [4, 4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.05882353, -128)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);

        let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

        tensor_3.dequantize().into_data().assert_approx_eq(
            &TensorData::from([
                [14., 38., 62., 86.],
                [38., 126., 214., 302.],
                [62., 214., 366., 518.],
                [86., 302., 518., 734.],
            ]),
            1,
        );
    }

    #[test]
    fn test_matmul_simple_2() {
        let device = Default::default();
        // Quantized [[1.0, 2.0, 3.0, 4.0]]
        let data = TensorData::quantized(
            vec![32i8, 64, 95, 127],
            [1, 4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.031496063)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[1.0, 2.0, 3.0, 4.0]]
        let data = TensorData::quantized(
            vec![64i8, 85, 106, 127],
            [4, 1],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected = TensorData::from([[50.0]]);

        tensor_3
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    fn test_matmul_simple_3() {
        let device = Default::default();
        // Quantized [[3., 3., 3.], [4., 4., 4.], [5., 5., 5.], [6., 6., 6.]]
        let data = TensorData::quantized(
            vec![64i8, 64, 64, 85, 85, 85, 106, 106, 106, 127, 127, 127],
            [4, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]]
        let data = TensorData::quantized(
            vec![32i8, 64, 95, 127, 32, 64, 95, 127, 32, 64, 95, 127],
            [3, 4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.031496063)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected = TensorData::from([
            [9., 18., 27., 36.],
            [12., 24., 36., 48.],
            [15., 30., 45., 60.],
            [18., 36., 54., 72.],
        ]);

        tensor_3
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_inner_dimensions_are_not_equal() {
        let device = Default::default();
        // Quantized [[3., 3.], [4., 4.], [5., 5.], [6., 6.]]
        let data = TensorData::quantized(
            vec![64i8, 64, 85, 85, 106, 106, 127, 127],
            [4, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]]
        let data = TensorData::quantized(
            vec![32i8, 64, 95, 127, 32, 64, 95, 127, 32, 64, 95, 127],
            [4, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.031496063)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let _ = tensor_1.matmul(tensor_2);
    }
}
