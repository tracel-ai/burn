#[burn_tensor_testgen::testgen(bernoulli)]
mod tests {
    use super::*;

    // Use the reexported test dependency.
    use burn_cubecl::tests::serial_test;
    use serial_test::serial;

    use core::f32;

    use burn_cubecl::kernel::prng::tests_utils::calculate_bin_stats;
    use burn_tensor::{backend::Backend, Distribution, Shape, Tensor};

    #[test]
    #[serial]
    fn number_of_1_proportional_to_prob() {
        TestBackend::seed(0);
        let shape: Shape = [40, 40].into();
        let device = Default::default();
        let prob = 0.7;

        let tensor_1 =
            Tensor::<TestBackend, 2>::random(shape.clone(), Distribution::Bernoulli(prob), &device);

        // High bound slightly over 1 so 1.0 is included in second bin
        let bin_stats = calculate_bin_stats(
            tensor_1
                .into_data()
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap(),
            2,
            0.,
            1.1,
        );
        assert!(
            f32::abs((bin_stats[1].count as f32 / shape.num_elements() as f32) - prob as f32)
                < 0.05
        );
    }

    #[test]
    #[serial]
    fn runs_test() {
        TestBackend::seed(0);
        let shape = Shape::new([512, 512]);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::random(shape, Distribution::Bernoulli(0.5), &device);

        let data = tensor.into_data();
        let numbers = data
            .as_slice::<<TestBackend as Backend>::FloatElem>()
            .unwrap();
        let stats = calculate_bin_stats(numbers, 2, 0., 1.1);
        let n_0 = stats[0].count as f32;
        let n_1 = stats[1].count as f32;
        let n_runs = (stats[0].n_runs + stats[1].n_runs) as f32;

        let expectation = (2. * n_0 * n_1) / (n_0 + n_1) + 1.0;
        let variance = ((2. * n_0 * n_1) * (2. * n_0 * n_1 - n_0 - n_1))
            / ((n_0 + n_1).powf(2.) * (n_0 + n_1 - 1.));
        let z = (n_runs - expectation) / f32::sqrt(variance);

        // below 2 means we can have good confidence in the randomness
        // we put 2.5 to make sure it passes even when very unlucky
        assert!(z.abs() < 2.5);
    }
}
