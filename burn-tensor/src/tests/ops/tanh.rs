#[burn_tensor_testgen::testgen(tanh)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_tanh_ops() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.tanh().into_data();

        let data_expected = Data::from([[0.0, 0.7615, 0.9640], [0.9950, 0.9993, 0.9999]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn tanh_should_not_have_numerical_bugs_on_macos() {
        fn tanh_one_value(input: f32) -> f32 {
            let tensor = Tensor::<TestBackend, 1>::ones([1]) * input;
            let output = tensor.tanh().into_primitive();
            Tensor::<TestBackend, 1>::from_primitive(output)
                .into_data()
                .value[0]
        }

        let ok = tanh_one_value(43.0); // metal tanh gives 1.0 which is the right answer
        let zero = tanh_one_value(44.0); // metal tanh gives zero when within 43.67..44.36
        let nan = tanh_one_value(45.0); // metal tanh gives nan when over 44.36

        assert!(!ok.is_nan() && ok == 1.0);
        assert!(!zero.is_nan() && zero == 1.0);
        assert!(!nan.is_nan() && nan == 1.0);
    }
}
