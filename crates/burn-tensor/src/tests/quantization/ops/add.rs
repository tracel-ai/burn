#[burn_tensor_testgen::testgen(q_add)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_add_d2() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default());
        // Quantized [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        let data = TensorData::quantized(
            vec![69i8, 81, 92, 104, 115, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.08661418)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor_1 + tensor_2;

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]), 1);
    }

    #[test]
    fn test_add_broadcast() {
        // Quantized [[0.0, 1.0, 2.0]]
        let data = TensorData::quantized(
            vec![0i8, 64, 127],
            [1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.015748031)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default());
        // Quantized [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
        let data = TensorData::quantized(
            vec![48i8, 64, 79, 95, 111, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.062992126)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor_1 + tensor_2;

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[3.0, 5.0, 7.0], [6.0, 8.0, 10.0]]), 1);
    }

    #[test]
    fn test_add_different_strides_rhs() {
        // Quantized [[0.0, 1.0], [2.0, 3.0]]
        let data = TensorData::quantized(
            vec![0i8, 42, 85, 127],
            [2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        // We need to execute an operation after `from data` to trigger inplace in some backends.
        // Which is the operation that might be problematic in this case.
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default()) * 1;
        // Quantized [[4.0, 5.0], [6.0, 7.0]]
        let data = TensorData::quantized(
            vec![73i8, 91, 109, 127],
            [2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &Default::default()) * 1;

        let output = tensor_1 + tensor_2.transpose();

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[4.0, 7.0], [7.0, 10.0]]), 1);
    }

    #[test]
    fn test_add_different_strides_lhs() {
        // Quantized [[0.0, 1.0], [2.0, 3.0]]
        let data = TensorData::quantized(
            vec![0i8, 42, 85, 127],
            [2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        // We need to execute an operation after `from data` to trigger inplace in some backends.
        // Which is the operation that might be problematic in this case.
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default()) * 1;
        // Quantized [[4.0, 5.0], [6.0, 7.0]]
        let data = TensorData::quantized(
            vec![73i8, 91, 109, 127],
            [2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.05511811)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &Default::default()) * 1;

        let output = tensor_1.transpose() + tensor_2;

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[4.0, 7.0], [7.0, 10.0]]), 1);
    }

    #[test]
    fn test_add_different_strides_broadcast() {
        // Quantized [[0.0, 1.0], [2.0, 3.0]]
        let data = TensorData::quantized(
            vec![0i8, 42, 85, 127],
            [2, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.023622047)),
        );
        // We need to execute an operation after `from data` to trigger inplace in some backends.
        // Which is the operation that might be problematic in this case.
        let tensor_1 = TestTensor::<2>::from_data(data, &Default::default()) * 1;
        // Quantized [[4.0, 5.0]]
        let data = TensorData::quantized(
            vec![102i8, 127],
            [1, 2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &Default::default()) * 1;

        let output = tensor_1.transpose() + tensor_2;

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[4.0, 7.0], [5.0, 8.0]]), 1);
    }

    #[test]
    fn should_support_add_scalar_ops() {
        let scalar = 2.0;
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor + scalar;

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]), 1);
    }
}
