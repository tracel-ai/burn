use super::*;
use burn_tensor::Tolerance;
use burn_tensor::ops::ConvOptions;
use burn_tensor::{Distribution, module};

#[test]
fn conv2d_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<4>::random([6, 16, 32, 32], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([12, 8, 3, 3], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([12], Distribution::Default, &device);

    let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = ConvOptions::new([2, 3], [2, 3], [2, 3], 2);

    let output = module::conv2d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv2d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

#[test]
fn conv2d_should_match_reference_backend_implicit() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<4>::random([4, 16, 6, 6], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([16, 16, 3, 3], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([16], Distribution::Default, &device);

    let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = ConvOptions::new([1, 1], [2, 2], [1, 1], 1);

    let output = module::conv2d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv2d(input_ref, weight_ref, Some(bias_ref), options);

    let tolerance = Tolerance::default();
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), tolerance);
}

/// Regression test for bias loader in new implicit GEMM
#[test]
fn conv2d_should_match_reference_backend_bias_regression() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<4>::random([1, 1, 1, 1], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([32, 1, 3, 3], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([32], Distribution::Default, &device);

    let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);

    let output = module::conv2d(input, weight, Some(bias), options.clone()).permute([0, 2, 3, 1]);
    let output_ref =
        module::conv2d(input_ref, weight_ref, Some(bias_ref), options).permute([0, 2, 3, 1]);

    let tolerance = Tolerance::default();
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), tolerance);
}

#[test]
fn conv2d_weight_backward_should_run() {
    // https://github.com/tracel-ai/burn/issues/4226#issuecomment-3911335769
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
    let x = TestTensor::<4>::random([1, 1, 1, 672], Distribution::Default, &device);
    // let x = x.permute([0, 3, 1, 2]);

    let output_grad = TestTensor::<4>::random([1, 168, 1, 1], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([168, 672, 1, 1], Distribution::Default, &device);

    let x_ref = TestTensor::<4>::from_data(x.to_data(), &ref_device);
    let output_grad_ref = TestTensor::<4>::from_data(output_grad.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);

    // Input shape [672, 1] and strides [672, 672] should be valid
    let output = module::conv2d_weight_backward(
        x.permute([0, 3, 1, 2]),
        weight,
        output_grad,
        options.clone(),
    );

    // Input shape [672, 1] and strides [672, 672] should be valid
    let output_ref = module::conv2d_weight_backward(
        x_ref.permute([0, 3, 1, 2]),
        weight_ref,
        output_grad_ref,
        options,
    );

    let tolerance = Tolerance::default();
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), tolerance);
}
