#[burn_tensor_testgen::testgen(exp)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_exp_ops() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.exp().into_data();

        let data_expected = Data::from([[1.0, 2.71830, 7.3891], [20.0855, 54.5981, 148.4132]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
