#[burn_tensor_testgen::testgen(remainder)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    /// From https://pytorch.org/docs/stable/generated/torch.remainder.html
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

    /// Also from https://pytorch.org/docs/stable/generated/torch.remainder.html
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

    #[test]
    fn should_be_zero() {
        let data = Data::from([0.0, 0.0, 0.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.remainder_scalar(3.5);

        let data_actual = output.into_data();
        let data_expected = Data::from([0.0, 0.0, 0.0]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_have_no_remainder() {
        let data = Data::from([-4.0, 4.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.remainder_scalar(4.0);

        let data_actual = output.into_data();
        let data_expected = Data::from([-0.0, 0.0]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_be_negative() {
        let data = Data::from([-7.0, -3.0, 2.0, 6.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.remainder_scalar(-2.5);

        let data_actual = output.into_data();
        let data_expected = Data::from([-2.0, -0.50, -0.50, -1.5]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_fp_dividends() {
        let data = Data::from([-7.5, -2.5, 2.5, 7.5]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.remainder_scalar(3.0);

        let data_actual = output.into_data();
        let data_expected = Data::from([1.5, 0.5, 2.5, 1.5]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_large_divisor() {
        let data = Data::from([-1.0, 1.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.remainder_scalar(10.0);

        let data_actual = output.into_data();
        let data_expected = Data::from([9.0, 1.0]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_remainder_op() {
        let data = Data::from([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor % 2.0;

        let data_actual = output.into_data();
        let data_expected = Data::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
