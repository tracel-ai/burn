#[burn_tensor_testgen::testgen(normal)]
mod tests {
    use super::*;
    use burn_jit::kernel::prng::tests_utils::calculate_bin_stats;
    use burn_tensor::{backend::Backend, Distribution, Shape, Tensor, TensorData};
    use serial_test::serial;

    #[test]
    #[serial]
    fn empirical_mean_close_to_expectation() {
        TestBackend::seed(0);
        let shape = [100, 100];
        let device = Default::default();
        let mean = 10.;
        let tensor =
            Tensor::<TestBackend, 2>::random(shape, Distribution::Normal(mean, 2.), &device);
        let empirical_mean = tensor.mean().into_data();
        empirical_mean.assert_approx_eq(&TensorData::from([mean as f32]), 1);
    }

    #[test]
    #[serial]
    fn normal_respects_68_95_99_rule() {
        // https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
        let shape: Shape = [1000, 1000].into();
        let device = Default::default();
        let mu = 0.;
        let s = 1.;
        let tensor =
            Tensor::<TestBackend, 2>::random(shape.clone(), Distribution::Normal(mu, s), &device)
                .into_data();

        let stats = calculate_bin_stats(
            tensor
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap(),
            6,
            (mu - 3. * s) as f32,
            (mu + 3. * s) as f32,
        );
        let assert_approx_eq = |count, percent| {
            let expected = percent * shape.num_elements() as f32 / 100.;
            assert!(f32::abs(count as f32 - expected) < 2000.);
        };
        assert_approx_eq(stats[0].count, 2.1);
        assert_approx_eq(stats[1].count, 13.6);
        assert_approx_eq(stats[2].count, 34.1);
        assert_approx_eq(stats[3].count, 34.1);
        assert_approx_eq(stats[4].count, 13.6);
        assert_approx_eq(stats[5].count, 2.1);
    }
}
