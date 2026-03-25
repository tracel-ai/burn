use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, TensorPrimitive, module, ops::ModuleOps};

#[test]
pub fn max_pool2d_with_indices_backward_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<4>::random([32, 32, 32, 32], Distribution::Default, &device);
    let grad_output = TestTensor::<4>::random([32, 32, 16, 16], Distribution::Default, &device);

    let tensor_ref = TestTensor::<4>::from_data(tensor.to_data(), &ref_device);
    let grad_output_ref = TestTensor::<4>::from_data(grad_output.to_data(), &ref_device);
    let kernel_size = [3, 3];
    let stride = [2, 2];
    let padding = [1, 1];
    let dilation = [1, 1];

    let (_, indices) = module::max_pool2d_with_indices(
        tensor.clone(),
        kernel_size,
        stride,
        padding,
        dilation,
        false,
    );
    let (_, indices_ref) = module::max_pool2d_with_indices(
        tensor_ref.clone(),
        kernel_size,
        stride,
        padding,
        dilation,
        false,
    );
    let grad = TestBackend::max_pool2d_with_indices_backward(
        tensor.into_primitive().tensor(),
        kernel_size,
        stride,
        padding,
        dilation,
        false,
        grad_output.into_primitive().tensor(),
        indices.into_primitive(),
    )
    .x_grad;
    let grad_ref = TestBackend::max_pool2d_with_indices_backward(
        tensor_ref.into_primitive().tensor(),
        kernel_size,
        stride,
        padding,
        dilation,
        false,
        grad_output_ref.into_primitive().tensor(),
        indices_ref.into_primitive(),
    )
    .x_grad;

    TestTensor::<4>::from_primitive(TensorPrimitive::Float(grad))
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TestTensor::<4>::from_primitive(TensorPrimitive::Float(grad_ref)).into_data(),
            Tolerance::default(),
        );
}
