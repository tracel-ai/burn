#[burn_tensor_testgen::testgen(random)]
mod tests {
    use super::*;
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn rand_default() {
        let tensor =
            Tensor::<TestBackend, 1>::random([20], Distribution::Default, &Default::default());

        // check that the tensor is within the range of [0..1) (1 is exclusive)
        tensor.into_data().assert_within_range(0.0..1.0);
    }

    #[test]
    fn rand_uniform() {
        let tensor = Tensor::<TestBackend, 1>::random(
            [20],
            Distribution::Uniform(4., 5.),
            &Default::default(),
        );

        tensor.into_data().assert_within_range(4.0..5.0);
    }

    #[test]
    fn rand_bernoulli() {
        let tensor = Tensor::<TestBackend, 1>::random(
            [20],
            Distribution::Bernoulli(1.),
            &Default::default(),
        );

        assert_eq!(tensor.into_data(), [1f32; 20].into());
    }
}
