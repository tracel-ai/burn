#[burn_tensor_testgen::testgen(random)]
mod tests {
    use super::*;
    use burn_tensor::{
        Distribution, ElementComparison, ElementConversion, Tensor, Tolerance, backend::Backend,
        cast::ToElement, tests::Float,
    };

    #[test]
    fn rand_default() {
        let tensor = TestTensor::<1>::random([20], Distribution::Default, &Default::default());

        // check that the tensor is within the range of [0..1) (1 is exclusive)
        // the conversion can ceil the value if `FloatType` is less precise than f32
        let low = 0.elem::<FloatType>();
        let high = 1.elem::<FloatType>();
        if FloatType::EPSILON.to_f32() > f32::EPSILON {
            tensor.into_data().assert_within_range_inclusive(low..=high);
        } else {
            tensor.into_data().assert_within_range(low..high);
        }
    }

    #[test]
    fn rand_uniform() {
        let tensor =
            TestTensor::<1>::random([20], Distribution::Uniform(4., 5.), &Default::default());
        let low = 4.elem::<FloatType>();
        let high = 5.elem::<FloatType>();

        if FloatType::EPSILON.to_f32() > f32::EPSILON {
            tensor.into_data().assert_within_range_inclusive(low..=high);
        } else {
            tensor.into_data().assert_within_range(low..high);
        }
    }

    #[test]
    fn rand_uniform_int() {
        let low = 0.;
        let high = 5.;

        let tensor = TestTensorInt::<1>::random(
            [100_000],
            Distribution::Uniform(low, high),
            &Default::default(),
        );

        type IntElem = <TestBackend as Backend>::IntElem;

        tensor
            .into_data()
            .assert_within_range::<IntElem>(low.elem()..high.elem());
    }

    #[test]
    fn rand_bernoulli() {
        let tensor =
            TestTensor::<1>::random([20], Distribution::Bernoulli(1.), &Default::default());

        assert_eq!(tensor.into_data(), [FloatType::new(1f32); 20].into());
    }

    #[test]
    #[ignore] // TODO: mark serial for backends that handle the same devices (e.g. fusion)?
    fn test_seed_reproducibility() {
        let device = Default::default();
        TestBackend::seed(&device, 42);
        let t1 = TestTensor::<1>::random([5], Distribution::Default, &device);
        TestBackend::seed(&device, 42);
        let t2 = TestTensor::<1>::random([5], Distribution::Default, &device);

        t1.into_data()
            .assert_approx_eq::<FloatType>(&t2.into_data(), Tolerance::default());
    }
}
