use super::*;
use burn_tensor::{Device, Distribution, Shape};
use burn_tensor::{ElementConversion, Tolerance};

use serial_test::serial;

use cubek::random::{assert_at_least_one_value_per_bin, assert_wald_wolfowitz_runs_test};

#[test]
#[serial]
fn values_all_within_interval_default() {
    let device = Device::default();
    device.seed(0);
    let shape = [24, 24];

    let tensor = TestTensor::<2>::random(shape, Distribution::Default, &device);
    tensor
        .to_data()
        .assert_within_range::<FloatElem>(0.elem()..1.elem());
}

#[test]
#[serial]
fn values_all_within_interval_uniform() {
    let device = Device::default();
    device.seed(0);
    let shape = [24, 24];

    let tensor = TestTensor::<2>::random(shape, Distribution::Uniform(5., 17.), &device);
    tensor
        .to_data()
        .assert_within_range::<FloatElem>(5.elem()..17.elem());
}

#[test]
#[serial]
fn at_least_one_value_per_bin_uniform() {
    let device = Device::default();
    device.seed(0);
    let shape = [64, 64];

    let tensor =
        TestTensor::<2>::random(shape, Distribution::Uniform(-5., 10.), &device).into_data();
    let numbers = tensor.as_slice::<FloatElem>().unwrap();

    assert_at_least_one_value_per_bin(numbers, 3, -5., 10.);
}

#[test]
#[serial]
fn runs_test() {
    let device = Device::default();
    device.seed(0);
    let shape = Shape::new([512, 512]);
    let tensor = TestTensor::<2>::random(shape, Distribution::Default, &device).into_data();

    let numbers = tensor.as_slice::<FloatElem>().unwrap();

    assert_wald_wolfowitz_runs_test(numbers, 0., 1.);
}

#[test]
#[serial]
fn int_values_all_within_interval_uniform() {
    let device = Device::default();
    device.seed(0);
    let shape = Shape::new([20, 20]);
    let tensor = TestTensorInt::<2>::random(shape, Distribution::Default, &device);

    let data_float = tensor.float().into_data();

    data_float.assert_within_range(0..255);
}

#[test]
#[serial]
fn at_least_one_value_per_bin_int_uniform() {
    let device = Device::default();
    device.seed(0);
    let shape = Shape::new([64, 64]);

    let tensor = TestTensorInt::<2>::random(shape, Distribution::Uniform(-10.0, 10.0), &device);

    let data_float = tensor.float().into_data();

    let numbers = data_float.as_slice::<FloatElem>().unwrap();

    assert_at_least_one_value_per_bin(numbers, 10, -10., 10.);
}

#[test]
fn should_not_fail_on_non_float_autotune() {
    let device = Device::default();
    let tensor_1 = TestTensor::<2>::from_data([[1., 2., 3.], [3., 4., 5.]], &device);

    // Autotune of all (reduce) on lower_equal_elem's output calls uniform distribution
    tensor_1.lower_equal_elem(1.0).all();
}

#[test]
#[serial]
fn test_seed_reproducibility() {
    let device = Device::default();
    device.seed(42);
    let t1 = TestTensor::<1>::random([5], Distribution::Default, &device);
    device.seed(42);
    let t2 = TestTensor::<1>::random([5], Distribution::Default, &device);

    t1.into_data()
        .assert_approx_eq::<FloatElem>(&t2.into_data(), Tolerance::default());
}
