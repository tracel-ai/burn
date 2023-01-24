#[burn_tensor_testgen::testgen(cos)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_cos_ops() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.cos().into_data();

        let data_expected = Data::from([[1.0, 0.5403, -0.4161], [-0.9899, -0.6536, 0.2836]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
