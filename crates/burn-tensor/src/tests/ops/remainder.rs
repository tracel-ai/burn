#[burn_tensor_testgen::testgen(remainder)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn should_support_remainder_basic() {
        let data = Data::from([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.remainder_scalar(2.0);

        let data_actual = output.into_data();
        let data_expected = Data::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_remainder_float() {
        let data = Data::from([1.0, 2.0, 3.0, 4.0, 5.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.remainder_scalar(-1.5);

        let data_actual = output.into_data();
        let data_expected = Data::from([-0.5, -1.0, 0.0, -0.5, -1.0]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
