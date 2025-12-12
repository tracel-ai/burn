use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, Tensor, module};

#[test]
fn conv2d_should_match_reference_backend() {
    let test_device = Default::default();
    let input =
        Tensor::<TestBackend, 4>::random([6, 16, 32, 32], Distribution::Default, &test_device);
    let weight =
        Tensor::<TestBackend, 4>::random([12, 8, 3, 3], Distribution::Default, &test_device);
    let bias = Tensor::<TestBackend, 1>::random([12], Distribution::Default, &test_device);
    let ref_device = Default::default();

    let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = burn_tensor::ops::ConvOptions::new([2, 3], [2, 3], [2, 3], 2);

    let output = module::conv2d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv2d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

#[test]
fn conv2d_should_match_reference_backend_implicit() {
    let test_device = Default::default();
    let input =
        Tensor::<TestBackend, 4>::random([4, 16, 6, 6], Distribution::Default, &test_device);
    let weight =
        Tensor::<TestBackend, 4>::random([16, 16, 3, 3], Distribution::Default, &test_device);
    let bias = Tensor::<TestBackend, 1>::random([16], Distribution::Default, &test_device);
    let ref_device = Default::default();

    let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = burn_tensor::ops::ConvOptions::new([1, 1], [2, 2], [1, 1], 1);

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
    let test_device = Default::default();
    let input = Tensor::<TestBackend, 4>::random([1, 1, 1, 1], Distribution::Default, &test_device);
    let weight =
        Tensor::<TestBackend, 4>::random([32, 1, 3, 3], Distribution::Default, &test_device);
    let bias = Tensor::<TestBackend, 1>::random([32], Distribution::Default, &test_device);
    let ref_device = Default::default();

    let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data(), &ref_device);

    let options = burn_tensor::ops::ConvOptions::new([1, 1], [1, 1], [1, 1], 1);

    let output = module::conv2d(input, weight, Some(bias), options.clone()).permute([0, 2, 3, 1]);
    let output_ref =
        module::conv2d(input_ref, weight_ref, Some(bias_ref), options).permute([0, 2, 3, 1]);

    let tolerance = Tolerance::default();
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), tolerance);
}
