#[burn_tensor_testgen::testgen(random)]
mod tests {
    use super::*;
    use burn_tensor::{Distribution, Tensor, cast::ToElement, tests::Float};

    #[test]
    fn rand_default() {
        let tensor = TestTensor::<1>::random([20], Distribution::Default, &Default::default());

        // check that the tensor is within the range of [0..1) (1 is exclusive)
        // the conversion can ceil the value if `FloatType` is less precise than f32
        if FloatType::EPSILON.to_f32() > f32::EPSILON {
            tensor.into_data().assert_within_range_inclusive(0.0..=1.0);
        } else {
            tensor.into_data().assert_within_range(0.0..1.0);
        }
    }

    #[test]
    fn rand_uniform() {
        let tensor =
            TestTensor::<1>::random([20], Distribution::Uniform(4., 5.), &Default::default());

        if FloatType::EPSILON.to_f32() > f32::EPSILON {
            tensor.into_data().assert_within_range_inclusive(4.0..=5.0);
        } else {
            tensor.into_data().assert_within_range(4.0..5.0);
        }
    }

    #[test]
    fn rand_bernoulli() {
        let tensor =
            TestTensor::<1>::random([20], Distribution::Bernoulli(1.), &Default::default());

        assert_eq!(tensor.into_data(), [FloatType::new(1f32); 20].into());
    }
}
