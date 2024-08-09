#[burn_tensor_testgen::testgen(q_abs)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_abs_ops() {
        // Quantized [[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]
        let data = TensorData::quantized(
            vec![0i8, -25, 51, 76, 102, -127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.abs();

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), 1);
    }
}
