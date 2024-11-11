#[burn_tensor_testgen::testgen(q_floor)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_floor_ops() {
        let data = TensorData::quantized(
            // [[24.0423, 87.9478, 76.1838], [59.6929, 43.8169, 94.8826]]
            vec![-63, 108, 76, 32, -10, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(
                0.3725856688608348,
                -128,
            )),
        );
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.floor();
        let expected = TensorData::from([[24., 87., 76.], [59., 43., 95.]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}
