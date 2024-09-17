#[burn_tensor_testgen::testgen(q_aggregation)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{
        AffineQuantization, QuantizationStrategy, SymmetricQuantization,
    };
    use burn_tensor::{Shape, Tensor, TensorData};

    #[test]
    fn test_should_mean() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.mean();

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([15.0 / 6.0]), 1);
    }

    #[test]
    fn test_should_sum() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.sum();

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([15.0]), 1);
    }

    #[test]
    fn test_should_mean_last_dim() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.mean_dim(1);
        let expected = TensorData::from([[3.0 / 3.0], [12.0 / 3.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn test_should_sum_last_dim() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.sum_dim(1);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[3.0], [12.0]]), 1);
    }

    #[test]
    fn test_should_sum_first_dim() {
        // Quantized [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]
        let data = TensorData::quantized(
            vec![95i8, 32, 64, 127, 64, 95],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.031496063)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.sum_dim(0);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[7.0, 3.0, 5.0]]), 1);
    }

    #[test]
    fn test_should_mean_first_dim() {
        // Quantized [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]
        let data = TensorData::quantized(
            vec![95i8, 32, 64, 127, 64, 95],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.031496063)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.mean_dim(0);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[7.0 / 2.0, 3.0 / 2.0, 5.0 / 2.0]]), 1);
    }

    #[test]
    fn test_should_sum_mid_dim_3d_non_contiguous_1() {
        // Quantized [
        //     [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
        //     [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
        // ]
        let data = TensorData::quantized(
            vec![36i8, 73, 18, 127, -91, 54, 54, 18, 36, 73, 36, 54],
            [2, 2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let output = tensor.swap_dims(0, 2).sum_dim(1);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq(
            &TensorData::new(vec![9.0, 7.0, -1.0, 3.0, 4.0, 5.0], [3, 1, 2]),
            1,
        );
    }

    #[test]
    fn test_should_sum_mid_dim_3d_non_contiguous_2() {
        // Quantized [
        //     [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
        //     [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
        // ]
        let data = TensorData::quantized(
            vec![36i8, 73, 18, 127, -91, 54, 54, 18, 36, 73, 36, 54],
            [2, 2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let output = tensor.swap_dims(0, 1).sum_dim(1);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq(
            &TensorData::new(vec![5.0, 5.0, 3.0, 11.0, -3.0, 6.0], [2, 1, 3]),
            1,
        );
    }

    #[test]
    fn test_prod_float() {
        // Quantized [[2.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        // NOTE: we use affine quantization to reduce quantization errors since `prod()` amplifies the error
        let data = TensorData::quantized(
            vec![-26i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());
        let output = tensor.prod();

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([240.0]), 3);

        // Quantized [[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![51i8, 0, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor_with_zero = TestTensor::<2>::from_data(data, &Default::default());
        let output = tensor_with_zero.prod();

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([0.0]), 5);
    }

    #[test]
    fn test_prod_dim_float() {
        // Quantized [[2.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        // NOTE: we use affine quantization to reduce quantization errors since `prod()` amplifies the error
        let data = TensorData::quantized(
            vec![-26i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());
        let output = tensor.prod_dim(1);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[4.0], [60.0]]), 1);

        // Quantized [[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-26i8, -128, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor_with_zero = TestTensor::<2>::from_data(data, &Default::default());
        let output = tensor_with_zero.prod_dim(1);
        let expected = TensorData::from([[0.0], [60.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
