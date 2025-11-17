#[burn_tensor_testgen::testgen(remainder)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    /// From https://pytorch.org/docs/stable/generated/torch.remainder.html
    #[test]
    fn should_support_remainder_basic() {
        let device = Default::default();
        let lhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
            &device,
        );
        let rhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([2.0, 3.0, 1.0, 2.0, 1.0, 3.0]),
            &device,
        );
        let output = lhs.remainder(rhs);
        let expected = TensorData::from([1.0, 1.0, -0.0, 1.0, 0.0, 0.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_remainder_basic_scalar() {
        let data = TensorData::from([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]);
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor.remainder_scalar(2.0);
        let expected = TensorData::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_remainder_float() {
        let device = Default::default();
        let lhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([1.0, 2.0, 3.0, 4.0, 5.0]),
            &device,
        );
        let rhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([1.4233, 2.7313, 0.2641, 1.9651, 0.5897]),
            &device,
        );
        let output = lhs.remainder(rhs);
        let expected = TensorData::from([1.0, 2.0, 0.0949, 0.0698, 0.2824]);

        // Metal has less precise remainder function
        let tolerance = Tolerance::default()
            .set_half_precision_relative(1e-2)
            .set_half_precision_absolute(2e-3);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, tolerance);
    }

    /// Also from https://pytorch.org/docs/stable/generated/torch.remainder.html
    #[test]
    fn should_support_remainder_float_scalar() {
        let data = TensorData::from([1.0, 2.0, 3.0, 4.0, 5.0]);
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor.clone().remainder_scalar(-1.5);
        let expected = TensorData::from([-0.5, -1.0, 0.0, -0.5, -1.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_be_zero() {
        let device = Default::default();
        let lhs = Tensor::<TestBackend, 1>::from_data(TensorData::from([0.0, 0.0, 0.0]), &device);
        let rhs = Tensor::<TestBackend, 1>::from_data(TensorData::from([3.5, -2.1, 1e-4]), &device);

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([0.0, 0.0, 0.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_be_zero_scalar() {
        let data = TensorData::from([0.0, 0.0, 0.0]);
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor.clone().remainder_scalar(3.5);
        let expected = TensorData::from([0.0, 0.0, 0.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_have_no_remainder() {
        let device = Default::default();
        let lhs = Tensor::<TestBackend, 1>::from_data(
            // Previous values failed on some vulkan backends (driver bug?)
            // TensorData::from([-1.4843, 1.1350, -2.1563, 1.0862, 0.5, 3.6587]),
            TensorData::from([-1.0, 1.5, -2.0, 2.5, 0.5, 4.0]),
            &device,
        );
        let rhs = Tensor::<TestBackend, 1>::from_data(
            // TensorData::from([1.4843, 1.1350, 2.1563, 1.0862, 0.5, 3.6587]),
            TensorData::from([1.0, 1.5, 2.0, 2.5, 0.5, 4.0]),
            &device,
        );

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([-0., 0., -0., 0., 0., 0.]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_have_no_remainder_scalar() {
        let data = TensorData::from([-4.0, 4.0]);
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor.remainder_scalar(4.0);
        let expected = TensorData::from([-0.0, 0.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_be_negative() {
        let device = Default::default();

        let lhs =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([-7.0, -3.0, 2.0, 6.0]), &device);
        let rhs = Tensor::<TestBackend, 1>::from_data(
            TensorData::from([-2.5, -2.1, -1.5, -3.25]),
            &device,
        );

        let output = lhs.remainder(rhs);
        let expected = TensorData::from([-2.0, -0.9, -1.0, -0.5]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_be_negative_scalar() {
        let data = TensorData::from([-7.0, -3.0, 2.0, 6.0]);
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor.clone().remainder_scalar(-2.5);
        let expected = TensorData::from([-2.0, -0.50, -0.50, -1.5]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_fp_dividends() {
        let data = TensorData::from([-7.5, -2.5, 2.5, 7.5]);
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor.remainder_scalar(3.0);
        let expected = TensorData::from([1.5, 0.5, 2.5, 1.5]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        // for tensor.remainder case, tests above have already covered float point dividend cases
    }

    #[test]
    fn should_support_large_divisor() {
        let device = Default::default();

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

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_large_divisor_scalar() {
        let data = TensorData::from([-1.0, 1.0]);
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor.remainder_scalar(10.0);
        let expected = TensorData::from([9.0, 1.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_remainder_op() {
        let device = Default::default();
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

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_remainder_scalar_op() {
        let data = TensorData::from([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]);
        let device = Default::default();
        let tensor = TestTensor::<1>::from_data(data, &device);

        let output = tensor % 2.0;
        let expected = TensorData::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_int_remainder_basic() {
        let data = TensorData::from([-3, -2, -1, 1, 2, 3]);
        let device = Default::default();
        let lhs = TestTensorInt::<1>::from_data(data, &device);

        let rhs = TestTensorInt::from_data(TensorData::from([2, 3, 1, 2, 1, 3]), &device);
        let output = lhs.remainder(rhs);
        let expected = TensorData::from([1, 1, -0, 1, 0, 0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_int_remainder_basic_scalar() {
        let data = TensorData::from([-3, -2, -1, 1, 2, 3]);
        let device = Default::default();
        let tensor = TestTensorInt::<1>::from_data(data, &device);

        let output = tensor.remainder_scalar(2);
        let expected = TensorData::from([1, 0, 1, 1, 0, 1]);

        output.into_data().assert_eq(&expected, false);
    }
}
