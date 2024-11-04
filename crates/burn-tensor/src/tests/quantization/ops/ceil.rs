#[burn_tensor_testgen::testgen(q_ceil)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_ceil_ops() {
        let data = TensorData::quantized(
            vec![-63i8, 108, 76, 32, -10, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(
                0.3725856688608348,
                -128,
            )),
        );
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.ceil();
        let expected = TensorData::from([[25., 88., 77.], [60., 44., 96.]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}
