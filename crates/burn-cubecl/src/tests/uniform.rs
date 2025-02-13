#[burn_tensor_testgen::testgen(uniform)]
mod tests {
    use super::*;
    use core::f32;

    use burn_tensor::{backend::Backend, ops::IntTensorOps, Distribution, Int, Shape, Tensor};

    use burn_cubecl::kernel::prng::tests_utils::calculate_bin_stats;
    use serial_test::serial;

    #[test]
    #[serial]
    fn values_all_within_interval_default() {
        TestBackend::seed(0);
        let shape = [24, 24];
        let device = Default::default();

        let tensor = Tensor::<TestBackend, 2>::random(shape, Distribution::Default, &device);
        tensor.to_data().assert_within_range(0..1);
    }

    #[test]
    #[serial]
    fn values_all_within_interval_uniform() {
        TestBackend::seed(0);
        let shape = [24, 24];
        let device = Default::default();

        let tensor =
            Tensor::<TestBackend, 2>::random(shape, Distribution::Uniform(5., 17.), &device);
        tensor.to_data().assert_within_range(5..17);
    }

    #[test]
    #[serial]
    fn at_least_one_value_per_bin_uniform() {
        TestBackend::seed(0);
        let shape = [64, 64];
        let device = Default::default();

        let tensor =
            Tensor::<TestBackend, 2>::random(shape, Distribution::Uniform(-5., 10.), &device)
                .into_data();
        let numbers = tensor
            .as_slice::<<TestBackend as Backend>::FloatElem>()
            .unwrap();
        let stats = calculate_bin_stats(numbers, 3, -5., 10.);
        assert!(stats[0].count >= 1);
        assert!(stats[1].count >= 1);
        assert!(stats[2].count >= 1);
    }

    #[test]
    #[serial]
    fn runs_test() {
        TestBackend::seed(0);
        let shape = Shape::new([512, 512]);
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 2>::random(shape, Distribution::Default, &device).into_data();

        let numbers = tensor
            .as_slice::<<TestBackend as Backend>::FloatElem>()
            .unwrap();
        let stats = calculate_bin_stats(numbers, 2, 0., 1.);
        let n_0 = stats[0].count as f32;
        let n_1 = stats[1].count as f32;
        let n_runs = (stats[0].n_runs + stats[1].n_runs) as f32;

        let expectation = (2. * n_0 * n_1) / (n_0 + n_1) + 1.0;
        let variance = ((2. * n_0 * n_1) * (2. * n_0 * n_1 - n_0 - n_1))
            / ((n_0 + n_1).powf(2.) * (n_0 + n_1 - 1.));
        let z = (n_runs - expectation) / variance.sqrt();

        // below 2 means we can have good confidence in the randomness
        // we put 2.5 to make sure it passes even when very unlucky
        assert!(z.abs() < 2.5);
    }

    #[test]
    #[serial]
    fn int_values_all_within_interval_uniform() {
        TestBackend::seed(0);
        let shape = Shape::new([20, 20]);
        let device = Default::default();
        let tensor: Tensor<TestBackend, 2, Int> =
            Tensor::random(shape, Distribution::Default, &device);

        let data_float = tensor.float().into_data();

        data_float.assert_within_range(0..255);
    }

    #[test]
    #[serial]
    fn at_least_one_value_per_bin_int_uniform() {
        TestBackend::seed(0);
        let shape = Shape::new([64, 64]);
        let device = Default::default();

        let tensor: Tensor<TestBackend, 2, Int> =
            Tensor::random(shape, Distribution::Uniform(-10.0, 10.0), &device);

        let data_float = tensor.float().into_data();

        let numbers = data_float
            .as_slice::<<TestBackend as Backend>::FloatElem>()
            .unwrap();
        let stats = calculate_bin_stats(numbers, 10, -10., 10.);
        assert!(stats[0].count >= 1);
        assert!(stats[1].count >= 1);
        assert!(stats[2].count >= 1);
    }

    #[test]
    fn should_not_fail_on_non_float_autotune() {
        let device = Default::default();
        let tensor_1 = Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [3., 4., 5.]], &device);

        // Autotune of all (reduce) on lower_equal_elem's output calls uniform distribution
        tensor_1.lower_equal_elem(1.0).all();
    }
}
