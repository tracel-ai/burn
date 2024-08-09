#[burn_tensor_testgen::testgen(q_flip)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn flip_float() {
        // Quantized [[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let flipped = tensor.clone().flip([0, 2]);
        let expected = TensorData::from([[[5., 4., 3.]], [[2., 1., 0.]]]);

        // Precision 1 to approximate de/quantization errors
        flipped
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);

        // Test with no flip
        let flipped = tensor.clone().flip([]);
        tensor.into_data().assert_eq(&flipped.into_data(), true);
    }

    #[test]
    #[should_panic]
    fn flip_duplicated_axes() {
        // Quantized [[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        // Test with a duplicated axis
        let _ = tensor.flip([0, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn flip_out_of_bound_axis() {
        // Quantized [[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 1, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        // Test with an out of bound axis
        let _ = tensor.clone().flip([3, 0, 1]);
    }
}
