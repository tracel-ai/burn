#[burn_tensor_testgen::testgen(remainder)]
mod tests {
    use super::*;
    use burn_tensor::{Int, Tensor, TensorData};

    /// From https://pytorch.org/docs/stable/generated/torch.remainder.html
    #[test]
    fn should_support_remainder_basic() {
        let data = TensorData::from([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.clone().remainder_scalar(2.0);
        let expected = TensorData::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

        output.into_data().assert_approx_eq(&expected, 3);

        let rhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([2.0, 3.0, 1.0, 2.0, 1.0, 3.0]),
            &device,
        );
        let output = tensor.remainder(rhs);
        let expected = TensorData::from([1.0, 1.0, -0.0, 1.0, 0.0, 0.0]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    /// Also from https://pytorch.org/docs/stable/generated/torch.remainder.html
    #[test]
    fn should_support_remainder_float() {
        let data = TensorData::from([1.0, 2.0, 3.0, 4.0, 5.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.clone().remainder_scalar(-1.5);
        let expected = TensorData::from([-0.5, -1.0, 0.0, -0.5, -1.0]);

        output.into_data().assert_approx_eq(&expected, 3);

        let rhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([1.4233, 2.7313, 0.2641, 1.9651, 0.5897]),
            &device,
        );
        let output = tensor.remainder(rhs);
        let expected = TensorData::from([1.0, 2.0, 0.0949, 0.0698, 0.2824]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn should_be_zero() {
        let data = TensorData::from([0.0, 0.0, 0.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.clone().remainder_scalar(3.5);
        let expected = TensorData::from([0.0, 0.0, 0.0]);

        output.into_data().assert_approx_eq(&expected, 3);

        let rhs = Tensor::<TestBackend, 1>::from_data(TensorData::from([3.5, -2.1, 1e-5]), &device);
        let output = tensor.remainder(rhs);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn should_have_no_remainder() {
        let data = TensorData::from([-4.0, 4.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.remainder_scalar(4.0);
        let expected = TensorData::from([-0.0, 0.0]);

        output.into_data().assert_approx_eq(&expected, 3);

        let lhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-1.4843, 1.1350, -2.1563, 1.0862, 0.5034, 3.6587]),
            &device,
        );
        let rhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([1.4843, 1.1350, 2.1563, 1.0862, 0.5034, 3.6587]),
            &device,
        );

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([-0., 0., -0., 0., 0., 0.]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn should_be_negative() {
        let data = TensorData::from([-7.0, -3.0, 2.0, 6.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.clone().remainder_scalar(-2.5);
        let expected = TensorData::from([-2.0, -0.50, -0.50, -1.5]);

        output.into_data().assert_approx_eq(&expected, 3);

        let rhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-2.5, -2.1, -1.5, -3.25]),
            &device,
        );

        let output = tensor.remainder(rhs);
        let expected = TensorData::from([-2.0, -0.9, -1.0, -0.5]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn should_support_fp_dividends() {
        let data = TensorData::from([-7.5, -2.5, 2.5, 7.5]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.remainder_scalar(3.0);
        let expected = TensorData::from([1.5, 0.5, 2.5, 1.5]);

        output.into_data().assert_approx_eq(&expected, 3);

        // for tensor.remainder case, tests above have already covered float point dividend cases
    }

    #[test]
    fn should_support_large_divisor() {
        let data = TensorData::from([-1.0, 1.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor.remainder_scalar(10.0);
        let expected = TensorData::from([9.0, 1.0]);

        output.into_data().assert_approx_eq(&expected, 3);

        let lhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-1.0, 1.0, -1.5, 1.5, -1.0, 1.0, -1.5, 1.5]),
            &device,
        );
        let rhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([10.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0]),
            &device,
        );
        let output = lhs.remainder(rhs);
        let expected = TensorData::from([9.0, 1.0, 8.5, 1.5, -1.0, -9.0, -1.5, -8.5]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn should_support_remainder_op() {
        let data = TensorData::from([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let output = tensor % 2.0;
        let expected = TensorData::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

        output.into_data().assert_approx_eq(&expected, 3);

        let lhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
            &device,
        );
        let rhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([2.0, 3.0, 1.0, 2.0, 1.0, 3.0]),
            &device,
        );

        let output = lhs % rhs;
        let expected = TensorData::from([1.0, 1.0, -0.0, 1.0, 0.0, 0.0]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}
