use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::module::max_pool1d;

#[test]
fn test_max_pool1d_simple() {
    let kernel_size = 4;
    let padding = 0;
    let stride = 1;
    let dilation = 1;

    let device = Default::default();
    let x = TestAutodiffTensor::from_floats(
        [[[0.9861, 0.5474, 0.4477, 0.0732, 0.3548, 0.8221]]],
        &device,
    )
    .require_grad();
    let x_grad_expected =
        TestAutodiffTensor::<3>::from_floats([[[1., 1., 0., 0., 0., 1.]]], &device);

    let output = max_pool1d(x.clone(), kernel_size, stride, padding, dilation);
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_with_dilation() {
    let kernel_size = 4;
    let padding = 0;
    let stride = 1;
    let dilation = 2;

    let device = Default::default();
    let x = TestAutodiffTensor::from_floats(
        [[[
            0.5388, 0.0676, 0.7122, 0.8316, 0.0653, 0.9154, 0.1536, 0.9089, 0.8016, 0.7518, 0.2073,
            0.0501, 0.8811, 0.5604, 0.5075, 0.4384, 0.9963, 0.9698, 0.4988, 0.2609, 0.3391, 0.2230,
            0.4610, 0.5365, 0.6880,
        ]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestAutodiffTensor::<3>::from_floats(
        [[[
            0., 0., 1., 0., 0., 3., 0., 1., 2., 1., 0., 0., 2., 0., 0., 0., 4., 4., 0., 0., 0., 0.,
            0., 0., 1.,
        ]]],
        &device,
    );

    let output = max_pool1d(x.clone(), kernel_size, stride, padding, dilation);
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_complex() {
    let kernel_size = 4;
    let padding = 0;
    let stride = 1;
    let dilation = 1;

    let device = Default::default();
    let x = TestAutodiffTensor::from_floats(
        [[[
            0.5388, 0.0676, 0.7122, 0.8316, 0.0653, 0.9154, 0.1536, 0.9089, 0.8016, 0.7518, 0.2073,
            0.0501, 0.8811, 0.5604, 0.5075, 0.4384, 0.9963, 0.9698, 0.4988, 0.2609, 0.3391, 0.2230,
            0.4610, 0.5365, 0.6880,
        ]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestAutodiffTensor::<3>::from_floats(
        [[[
            0., 0., 0., 2., 0., 4., 0., 2., 1., 0., 0., 0., 4., 0., 0., 0., 4., 1., 1., 0., 0., 0.,
            1., 1., 1.,
        ]]],
        &device,
    );

    let output = max_pool1d(x.clone(), kernel_size, stride, padding, dilation);
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}

#[test]
fn test_max_pool1d_complex_with_padding() {
    let kernel_size = 4;
    let padding = 2;
    let stride = 1;
    let dilation = 1;

    let device = Default::default();
    let x = TestAutodiffTensor::from_floats(
        [[[
            0.5388, 0.0676, 0.7122, 0.8316, 0.0653, 0.9154, 0.1536, 0.9089, 0.8016, 0.7518, 0.2073,
            0.0501, 0.8811, 0.5604, 0.5075, 0.4384, 0.9963, 0.9698, 0.4988, 0.2609, 0.3391, 0.2230,
            0.4610, 0.5365, 0.6880,
        ]]],
        &device,
    )
    .require_grad();
    let x_grad_expected = TestAutodiffTensor::<3>::from_floats(
        [[[
            1., 0., 1., 2., 0., 4., 0., 2., 1., 0., 0., 0., 4., 0., 0., 0., 4., 1., 1., 0., 0., 0.,
            1., 1., 3.,
        ]]],
        &device,
    );

    let output = max_pool1d(x.clone(), kernel_size, stride, padding, dilation);
    let grads = output.backward();

    // Asserts
    let x_grad_actual = x.grad(&grads).unwrap();
    x_grad_expected
        .to_data()
        .assert_approx_eq::<FloatElem>(&x_grad_actual.to_data(), Tolerance::default());
}
