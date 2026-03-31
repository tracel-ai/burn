use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, TensorPrimitive, backend::Backend, module, ops::ModuleOps};

#[test]
fn avg_pool2d_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<4>::random([32, 32, 32, 32], Distribution::Default, &device);
    let tensor_ref = TestTensor::<4>::from_data(tensor.to_data(), &ref_device);
    let kernel_size = [3, 4];
    let stride = [1, 2];
    let padding = [1, 2];
    let count_include_pad = true;

    let pooled = module::avg_pool2d(
        tensor,
        kernel_size,
        stride,
        padding,
        count_include_pad,
        false,
    );
    let pooled_ref = module::avg_pool2d(
        tensor_ref,
        kernel_size,
        stride,
        padding,
        count_include_pad,
        false,
    );

    pooled
        .into_data()
        .assert_approx_eq::<FloatElem>(&pooled_ref.into_data(), Tolerance::default());
}

#[test]
fn avg_pool2d_backward_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    TestBackend::seed(&device, 0);
    TestBackend::seed(&ref_device, 0);

    let tensor = TestTensor::<4>::random([32, 32, 32, 32], Distribution::Default, &device);
    let tensor_ref = TestTensor::<4>::from_data(tensor.to_data(), &ref_device);
    let kernel_size = [3, 3];
    let stride = [1, 1];
    let padding = [1, 1];
    let count_include_pad = true;

    let shape_out = module::avg_pool2d(
        tensor.clone(),
        kernel_size,
        stride,
        padding,
        count_include_pad,
        false,
    )
    .shape();
    let grad_output = TestTensor::<4>::random(shape_out, Distribution::Default, &device);
    let grad_output_ref = TestTensor::<4>::from_data(grad_output.to_data(), &ref_device);

    let grad =
        TestTensor::<4>::from_primitive(TensorPrimitive::Float(TestBackend::avg_pool2d_backward(
            tensor.into_primitive().tensor(),
            grad_output.into_primitive().tensor(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            false,
        )));
    let grad_ref =
        TestTensor::<4>::from_primitive(TensorPrimitive::Float(TestBackend::avg_pool2d_backward(
            tensor_ref.into_primitive().tensor(),
            grad_output_ref.into_primitive().tensor(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            false,
        )));

    grad.into_data()
        .assert_approx_eq::<FloatElem>(&grad_ref.into_data(), Tolerance::default());
}
