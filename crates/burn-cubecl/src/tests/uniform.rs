#[burn_tensor_testgen::testgen(uniform)]
mod tests {
    use super::*;
    use core::f32;

    use burn_tensor::{Distribution, Int, Shape, Tensor, backend::Backend, ops::IntTensorOps};
    use burn_tensor::{ElementConversion, Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    use serial_test::serial;

    use cubecl::random::{assert_at_least_one_value_per_bin, assert_wald_wolfowitz_runs_test};

    #[test]
    #[serial]
    fn values_all_within_interval_default() {
        let device = Default::default();
        TestBackend::seed(&device, 0);
        let shape = [24, 24];

        let tensor = Tensor::<TestBackend, 2>::random(shape, Distribution::Default, &device);
        tensor
            .to_data()
            .assert_within_range::<FT>(0.elem()..1.elem());
    }

    #[test]
    #[serial]
    fn values_all_within_interval_uniform() {
        let device = Default::default();
        TestBackend::seed(&device, 0);
        let shape = [24, 24];

        let tensor =
            Tensor::<TestBackend, 2>::random(shape, Distribution::Uniform(5., 17.), &device);
        tensor
            .to_data()
            .assert_within_range::<FT>(5.elem()..17.elem());
    }

    #[test]
    #[serial]
    fn at_least_one_value_per_bin_uniform() {
        let device = Default::default();
        TestBackend::seed(&device, 0);
        let shape = [64, 64];

        let tensor =
            Tensor::<TestBackend, 2>::random(shape, Distribution::Uniform(-5., 10.), &device)
                .into_data();
        let numbers = tensor.as_slice::<FT>().unwrap();

        assert_at_least_one_value_per_bin(numbers, 3, -5., 10.);
    }

    #[test]
    #[serial]
    fn runs_test() {
        let device = Default::default();
        TestBackend::seed(&device, 0);
        let shape = Shape::new([512, 512]);
        let tensor =
            Tensor::<TestBackend, 2>::random(shape, Distribution::Default, &device).into_data();

        let numbers = tensor.as_slice::<FT>().unwrap();

        assert_wald_wolfowitz_runs_test(numbers, 0., 1.);
    }

    #[test]
    #[serial]
    fn int_values_all_within_interval_uniform() {
        let device = Default::default();
        TestBackend::seed(&device, 0);
        let shape = Shape::new([20, 20]);
        let tensor: Tensor<TestBackend, 2, Int> =
            Tensor::random(shape, Distribution::Default, &device);

        let data_float = tensor.float().into_data();

        data_float.assert_within_range(0..255);
    }

    #[test]
    #[serial]
    fn at_least_one_value_per_bin_int_uniform() {
        let device = Default::default();
        TestBackend::seed(&device, 0);
        let shape = Shape::new([64, 64]);

        let tensor: Tensor<TestBackend, 2, Int> =
            Tensor::random(shape, Distribution::Uniform(-10.0, 10.0), &device);

        let data_float = tensor.float().into_data();

        let numbers = data_float.as_slice::<FT>().unwrap();

        assert_at_least_one_value_per_bin(numbers, 10, -10., 10.);
    }

    #[test]
    fn should_not_fail_on_non_float_autotune() {
        let device = Default::default();
        let tensor_1 = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [3., 4., 5.]], &device);

        // Autotune of all (reduce) on lower_equal_elem's output calls uniform distribution
        tensor_1.lower_equal_elem(1.0).all();
    }
}
