#[burn_tensor_testgen::testgen(q_round)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{AffineQuantization, QuantizationStrategy};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_round_ops() {
        let data = TensorData::quantized(
            vec![-63, 108, 76, 32, -10, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(
                0.3725856688608348,
                -128,
            )),
        );
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.round();
        let expected = TensorData::from([[24., 88., 76.], [60., 44., 95.]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn should_round_ties_even() {
        let data = TensorData::quantized(
            vec![-69i8, -30, 9, 48, 87, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(
                0.02552864282968089,
                -128,
            )),
        );
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.round();
        let expected = TensorData::from([[2., 3., 3.], [4., 5., 7.]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}
