#[burn_tensor_testgen::testgen(q_div)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_div_ops() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![25i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output = tensor_1 / tensor_2;
        let expected = TensorData::from([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn test_div_broadcast() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0]]
        let data = TensorData::quantized(
            vec![0i8, 64, 127],
            [1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.015748031)),
        );
        let tensor_1 = TestTensor::<2>::from_data(data, &device);
        // Quantized [[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![25i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor_2 = TestTensor::<2>::from_data(data, &device);

        let output = tensor_1 / tensor_2;

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[0.0, 1.0, 1.0], [0.0, 0.25, 0.4]]), 1);
    }

    #[test]
    fn should_support_div_scalar_ops() {
        let scalar = 2.0;
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);

        let output = tensor / scalar;

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]]), 1);
    }
}
