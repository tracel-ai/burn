#[burn_tensor_testgen::testgen(log)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_support_log_ops() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let output = tensor.log();
        let expected = TensorData::from([
            [-f32::INFINITY, 0.0, core::f32::consts::LN_2],
            [1.0986, 1.3862, 1.6094],
        ]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}
