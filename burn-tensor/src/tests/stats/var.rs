#[burn_tensor_testgen::testgen(var)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Data, Tensor};

    type FloatElem = <TestBackend as Backend>::FloatElem;
    type IntElem = <TestBackend as Backend>::IntElem;

    #[test]
    fn test_var() {
        let tensor = TestTensor::from_data_devauto([[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]]);

        let data_actual = tensor.var(1).into_data();

        let data_expected = Data::from([[2.4892], [15.3333]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn test_var_mean() {
        let tensor = TestTensor::from_data_devauto([[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]]);

        let (var, mean) = tensor.var_mean(1);

        let var_expected = Data::from([[2.4892], [15.3333]]);
        let mean_expected = Data::from([[0.125], [1.]]);

        var_expected.assert_approx_eq(&(var.into_data()), 3);
        mean_expected.assert_approx_eq(&(mean.into_data()), 3);
    }

    #[test]
    fn test_var_bias() {
        let tensor = TestTensor::from_data_devauto([[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]]);

        let data_actual = tensor.var_bias(1).into_data();

        let data_expected = Data::from([[1.86688], [11.5]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn test_var_mean_bias() {
        let tensor = TestTensor::from_data_devauto([[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]]);

        let (var, mean) = tensor.var_mean_bias(1);

        let var_expected = Data::from([[1.86688], [11.5]]);
        let mean_expected = Data::from([[0.125], [1.]]);

        var_expected.assert_approx_eq(&(var.into_data()), 3);
        mean_expected.assert_approx_eq(&(mean.into_data()), 3);
    }
}
