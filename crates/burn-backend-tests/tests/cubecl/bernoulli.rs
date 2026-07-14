use super::*;

use serial_test::serial;

use core::f32;

use burn_tensor::{Device, Distribution, Shape};

use cubek::random::{assert_number_of_1_proportional_to_prob, assert_wald_wolfowitz_runs_test};

#[test]
#[serial]
fn number_of_1_proportional_to_prob() {
    let device = Device::default();
    device.seed(0);

    let shape: Shape = [40, 40].into();
    let prob = 0.7;

    let tensor =
        TestTensor::<2>::random(shape.clone(), Distribution::Bernoulli(prob), &device).into_data();

    let numbers = tensor.as_slice::<FloatElem>().unwrap();

    assert_number_of_1_proportional_to_prob(numbers, prob as f32);
}

#[test]
#[serial]
fn wald_wolfowitz_runs_test() {
    let device = Device::default();
    device.seed(0);

    let shape = Shape::new([512, 512]);
    let device = Device::default();
    let tensor = TestTensor::<2>::random(shape, Distribution::Bernoulli(0.5), &device);

    let data = tensor.into_data();
    let numbers = data.as_slice::<FloatElem>().unwrap();

    // High bound slightly over 1 so 1.0 is included in second bin
    assert_wald_wolfowitz_runs_test(numbers, 0., 1.1);
}
