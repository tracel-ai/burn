use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, module};

#[test]
pub fn max_pool2d_should_match_reference_backends() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<4>::random([32, 32, 32, 32], Distribution::Default, &device);
    let tensor_ref = TestTensor::<4>::from_data(tensor.to_data(), &ref_device);
    let kernel_size = [3, 3];
    let stride = [2, 2];
    let padding = [1, 1];
    let dilation = [1, 1];

    let pooled = module::max_pool2d(tensor, kernel_size, stride, padding, dilation, false);
    let pooled_ref = module::max_pool2d(tensor_ref, kernel_size, stride, padding, dilation, false);

    pooled
        .into_data()
        .assert_approx_eq::<FloatElem>(&pooled_ref.into_data(), Tolerance::default());
}

#[test]
pub fn max_pool2d_with_indices_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<4>::random([32, 32, 32, 32], Distribution::Default, &device);
    let tensor_ref = TestTensor::<4>::from_data(tensor.to_data(), &ref_device);
    let kernel_size = [3, 3];
    let stride = [2, 2];
    let padding = [1, 1];
    let dilation = [1, 1];

    let (pooled, indices) =
        module::max_pool2d_with_indices(tensor, kernel_size, stride, padding, dilation, false);
    let (pooled_ref, indices_ref) =
        module::max_pool2d_with_indices(tensor_ref, kernel_size, stride, padding, dilation, false);

    pooled
        .into_data()
        .assert_approx_eq::<FloatElem>(&pooled_ref.into_data(), Tolerance::default());
    indices
        .into_data()
        .assert_eq(&indices_ref.into_data(), false);
}
