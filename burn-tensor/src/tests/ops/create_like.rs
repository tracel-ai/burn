#[burn_tensor_testgen::testgen(create_like)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Distribution, Tensor};

    #[test]
    fn should_support_zeros_like() {
        let tensor = TestTensor::from_floats_devauto([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ]);

        let data_actual = tensor.zeros_like().into_data();

        let data_expected =
            Data::from([[[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]]]);

        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_ones_like() {
        let tensor = TestTensor::from_floats_devauto([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ]);

        let data_actual = tensor.ones_like().into_data();

        let data_expected =
            Data::from([[[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]]]);

        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_randoms_like() {
        let tensor = TestTensor::from_floats_devauto([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ]);

        let data_actual = tensor
            .random_like(Distribution::Uniform(0.99999, 1.))
            .into_data();

        let data_expected =
            Data::from([[[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]]]);

        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
