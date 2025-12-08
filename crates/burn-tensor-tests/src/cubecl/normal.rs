use super::*;
use burn_tensor::{Distribution, Shape, Tensor, backend::Backend};
use cubecl::random::{assert_mean_approx_equal, assert_normal_respects_68_95_99_rule};
use serial_test::serial;

#[test]
#[serial]
fn empirical_mean_close_to_expectation() {
    let device = Default::default();
    TestBackend::seed(&device, 0);

    let shape = [100, 100];
    let mean = 10.;
    let tensor = Tensor::<TestBackend, 2>::random(shape, Distribution::Normal(mean, 2.), &device)
        .into_data();
    let numbers = tensor.as_slice::<FloatElem>().unwrap();

    assert_mean_approx_equal(numbers, mean as f32);
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

    let numbers = tensor.as_slice::<FloatElem>().unwrap();

    assert_normal_respects_68_95_99_rule(numbers, mu as f32, s as f32);
}
