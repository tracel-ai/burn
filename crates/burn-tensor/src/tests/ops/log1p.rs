#[burn_tensor_testgen::testgen(log1p)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_exp_log1p() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.log1p();
        let expected = TensorData::from([
            [0.0, core::f32::consts::LN_2, 1.0986],
            [1.3862, 1.6094, 1.7917],
        ]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}
