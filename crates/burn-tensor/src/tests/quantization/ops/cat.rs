#[burn_tensor_testgen::testgen(q_cat)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Bool, Int, Tensor, TensorData};

    #[test]
    fn should_support_cat_ops_2d_dim0() {
        let device = Default::default();
        // Quantized [[1.0, 2.0, 3.0]]
        let data = TensorData::quantized(
            vec![42i8, 85, 127],
            [1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[4.0, 5.0, 6.0]]
        let data = TensorData::quantized(
            vec![85i8, 106, 127],
            [1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output = TestTensor::cat(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_cat_ops_2d_dim1() {
        let device = Default::default();
        // Quantized [[1.0, 2.0, 3.0]]
        let data = TensorData::quantized(
            vec![42i8, 85, 127],
            [1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[4.0, 5.0, 6.0]]
        let data = TensorData::quantized(
            vec![85i8, 106, 127],
            [1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output = TestTensor::cat(vec![tensor_1, tensor_2], 1);
        let expected = TensorData::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_cat_ops_3d() {
        let device = Default::default();
        // Quantized [[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]]]
        let data = TensorData::quantized(
            vec![41i8, 82, 123, 45, 86, 127],
            [2, 1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.024409449)),
        );
        let tensor_1 = TestTensor::<3>::from_data(data, &device);
        // Quantized [[4.0, 5.0, 6.0]]
        let data = TensorData::quantized(
            vec![85i8, 106, 127],
            [1, 1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let tensor_2 = TestTensor::<3>::from_data(data, &device);

        let output = TestTensor::cat(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([[[1.0, 2.0, 3.0]], [[1.1, 2.1, 3.1]], [[4.0, 5.0, 6.0]]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_dimensions_are_not_the_same() {
        let device = Default::default();
        // Quantized [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        let data = TensorData::quantized(
            vec![42i8, 85, 127, 42, 85, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[4.0, 5.0]]
        let data = TensorData::quantized(
            vec![102i8, 127],
            [2, 1],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);
        let tensor_1 = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], &device);
        let tensor_2 = TestTensor::from_data([[4.0, 5.0]], &device);

        let output = TestTensor::cat(vec![tensor_1, tensor_2], 0);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_cat_exceeds_dimension() {
        let device = Default::default();
        // Quantized [[1.0, 2.0, 3.0]]
        let data = TensorData::quantized(
            vec![42i8, 85, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[4.0, 5.0, 6.0]]
        let data = TensorData::quantized(
            vec![85i8, 106, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output = TestTensor::cat(vec![tensor_1, tensor_2], 3);
    }
}
