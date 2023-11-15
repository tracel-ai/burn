#[burn_tensor_testgen::testgen(cov)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Data, Tensor};

    type FloatElem = <TestBackend as Backend>::FloatElem;
    type IntElem = <TestBackend as Backend>::IntElem;

    #[test]
    fn test_cov_1() {
        let data = Data::from([[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.cov(1, 1).into_data();

        let data_expected = Data::from([[2.4892, -1.7333], [-1.7333, 15.3333]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn test_cov_4() {
        let data = Data::from([[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.cov(1, 0).into_data();

        let data_expected = Data::from([[1.8668, -1.2999], [-1.2999, 11.5]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn test_cov_2() {
        let data = Data::from([[0.5, 1.8], [0.2, -2.0], [3.0, -4.0], [5.0, 0.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.cov(1, 1).into_data();

        let data_expected = Data::from([
            [0.845, -1.43, -4.55, -3.25],
            [-1.43, 2.42, 7.7, 5.5],
            [-4.55, 7.7, 24.5, 17.5],
            [-3.25, 5.5, 17.5, 12.5],
        ]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn test_cov_3() {
        let data = Data::from([
            [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
            [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
            [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
            [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
        ]);
        let tensor = Tensor::<TestBackend, 3>::from_data(data);
        let data_actual = tensor.cov(0, 1).into_data();
        let data_expected = Tensor::<TestBackend, 3>::zeros([4, 4, 4]).to_data();
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
