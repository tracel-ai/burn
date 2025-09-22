#[burn_tensor_testgen::testgen(fmod)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_support_fmod_ops() {
        let dividend = TensorData::from([[5.3, -5.3], [7.5, -7.5]]);
        let divisor = TensorData::from([[2.0, 2.0], [3.0, 3.0]]);

        let dividend_tensor = TestTensor::<2>::from_data(dividend, &Default::default());
        let divisor_tensor = TestTensor::<2>::from_data(divisor, &Default::default());

        let output = dividend_tensor.fmod(divisor_tensor);
        let expected = TensorData::from([[1.3, -1.3], [1.5, -1.5]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_fmod_scalar() {
        let data = TensorData::from([5.3, -5.3, 7.5, -7.5]);
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.fmod_scalar(2.0);
        let expected = TensorData::from([1.3, -1.3, 1.5, -1.5]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_handle_positive_dividend_positive_divisor() {
        let dividend = TensorData::from([10.0, 7.5, 3.8, 1.2]);
        let divisor = TensorData::from([3.0, 2.0, 1.5, 0.7]);

        let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
        let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

        let output = dividend_tensor.fmod(divisor_tensor);
        let expected = TensorData::from([1.0, 1.5, 0.8, 0.5]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_handle_negative_dividend() {
        let dividend = TensorData::from([-10.0, -7.5, -3.8, -1.2]);
        let divisor = TensorData::from([3.0, 2.0, 1.5, 0.7]);

        let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
        let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

        let output = dividend_tensor.fmod(divisor_tensor);
        let expected = TensorData::from([-1.0, -1.5, -0.8, -0.5]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_handle_mixed_signs() {
        let dividend = TensorData::from([5.3, -5.3, 5.3, -5.3]);
        let divisor = TensorData::from([2.0, 2.0, -2.0, -2.0]);

        let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
        let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

        let output = dividend_tensor.fmod(divisor_tensor);
        // fmod result has same sign as dividend
        let expected = TensorData::from([1.3, -1.3, 1.3, -1.3]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
