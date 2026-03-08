use super::*;
use burn_tensor::Tolerance;
use burn_tensor::ops::{ConvOptions, ModuleOps};
use burn_tensor::{Distribution, Tensor, TensorPrimitive, module};

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

    let options = ConvOptions::new([2, 3], [2, 3], [2, 3], 2);

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
    let test_device = Default::default();
    let input = Tensor::<TestBackend, 4>::random([1, 1, 1, 1], Distribution::Default, &test_device);
    let weight =
        Tensor::<TestBackend, 4>::random([32, 1, 3, 3], Distribution::Default, &test_device);
    let bias = Tensor::<TestBackend, 1>::random([32], Distribution::Default, &test_device);
    let ref_device = Default::default();

    let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data(), &ref_device);
    let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = Tensor::<ReferenceBackend, 1>::from_data(bias.to_data(), &ref_device);

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
    let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
    let x = Tensor::<TestBackend, 4>::random([1, 1, 1, 672], Distribution::Default, &device);
    // let x = x.permute([0, 3, 1, 2]);

    let output_grad =
        Tensor::<TestBackend, 4>::random([1, 168, 1, 1], Distribution::Default, &device);
    let weight = Tensor::<TestBackend, 4>::random([168, 672, 1, 1], Distribution::Default, &device);

    let ref_device = Default::default();
    let x_ref = Tensor::<ReferenceBackend, 4>::from_data(x.to_data(), &ref_device);
    let output_grad_ref =
        Tensor::<ReferenceBackend, 4>::from_data(output_grad.to_data(), &ref_device);
    let weight_ref = Tensor::<ReferenceBackend, 4>::from_data(weight.to_data(), &ref_device);

    // Input shape [672, 1] and strides [672, 672] should be valid
    let output = TestBackend::conv2d_weight_backward(
        x.permute([0, 3, 1, 2]).into_primitive().tensor(),
        weight.into_primitive().tensor(),
        output_grad.into_primitive().tensor(),
        options.clone(),
    );

    // Input shape [672, 1] and strides [672, 672] should be valid
    let output_ref = ReferenceBackend::conv2d_weight_backward(
        x_ref.permute([0, 3, 1, 2]).into_primitive().tensor(),
        weight_ref.into_primitive().tensor(),
        output_grad_ref.into_primitive().tensor(),
        options,
    );

    let tolerance = Tolerance::default();
    Tensor::<TestBackend, 4>::from_primitive(TensorPrimitive::Float(output))
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &Tensor::<ReferenceBackend, 4>::from_primitive(TensorPrimitive::Float(output_ref))
                .into_data(),
            tolerance,
        );
}
