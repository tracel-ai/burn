#[burn_tensor_testgen::testgen(q_reshape)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{
        AffineQuantization, QuantizationStrategy, SymmetricQuantization,
    };
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_reshape_1d() {
        // Quantized [0.0, 1.0, 2.0, 3.0]
        let data = TensorData::quantized(
            vec![-128i8, -43, 42, 127],
            [4],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.011764706, -128)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.clone().reshape([1, 4]);
        let expected = TensorData::from([[0.0, 1.0, 2.0, 3.0]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_reshape_2d() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.clone().reshape([6]);
        let expected = TensorData::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_dim_infererence() {
        // Quantized [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        let data = TensorData::quantized(
            vec![-128i8, -105, -82, -58, -35, -12, 11, 34, 57, 81, 104, 127],
            [4, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.043137256, -128)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        // Infer the dimension via -1
        let reshaped = tensor.clone().reshape([2, -1]);
        assert_eq!(reshaped.shape(), [2, 6].into());

        // Infer the dimension via 0 (keep from the source) and -1 (infer)
        let reshaped = reshaped.reshape([0, 2, -1]);
        assert_eq!(reshaped.shape(), [2, 2, 3].into());

        // This is effectively as if we did a flatten
        let reshaped = tensor.clone().reshape([-1]);
        assert_eq!(reshaped.shape(), [12].into());

        // Keeping the first dimension the same (using 0)
        let reshaped = tensor.clone().reshape([0, 3]);
        assert_eq!(reshaped.shape(), [4, 3].into());
    }

    #[test]
    fn should_not_corrupt_after_slice() {
        // Quantized [0.0, 0.0]
        let data = TensorData::quantized(
            vec![0i8, 0],
            [2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.1)),
        );
        let zeros = TestTensor::<1>::from_data(data, &Default::default());
        zeros.clone().slice([1..2]).reshape([1]).exp();

        // May lead to zeroes being equal to [0.0, 1.0]
        zeros.dequantize().into_data().assert_eq(
            &Tensor::<TestBackend, 1>::zeros([2], &Default::default()).to_data(),
            true,
        );
    }

    #[test]
    #[should_panic]
    fn multiple_neg_ones() {
        // Quantized [0.0, 1.0, 2.0]
        let data = TensorData::quantized(
            vec![0i8, 64, 127],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.015748031)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let data_actual = tensor.reshape([-1, -1]).into_data();
    }

    #[test]
    #[should_panic]
    fn neg_value() {
        // Quantized [0.0, 1.0, 2.0]
        let data = TensorData::quantized(
            vec![0i8, 64, 127],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.015748031)),
        );
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let data_actual = tensor.reshape([-2, -1]).into_data();
    }
}
