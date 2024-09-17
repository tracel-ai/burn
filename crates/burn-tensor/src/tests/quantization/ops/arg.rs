#[burn_tensor_testgen::testgen(q_arg)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_argmax_2d_dim0() {
        // Quantized [[10.0, 11.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![115i8, 127, 23, 35, 46, 58],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.08661418)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.argmax(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([[0, 0, 1]]), false);
    }

    #[test]
    fn test_argmin_2d_dim0() {
        // Quantized [[10.0, 11.0, 2.0], [30.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![42i8, 47, 8, 127, 17, 21],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.23622048)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.argmin(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([[0, 1, 0]]), false);
    }

    #[test]
    fn test_argmax_2d_dim1() {
        // Quantized [[10.0, 11.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![115i8, 127, 23, 35, 46, 58],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.08661418)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.argmax(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1], [2]]), false);
    }

    #[test]
    fn test_argmin_2d_dim1() {
        // Quantized [[10.0, 11.0, 2.0], [30.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![42i8, 47, 8, 127, 17, 21],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.23622048)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.argmin(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[2], [1]]), false);
    }
}
