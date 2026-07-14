use super::*;
use burn_tensor::Device;
use burn_tensor::Distribution;
use burn_tensor::Tolerance;

#[test]
fn cat_should_match_reference_backend_dim0() {
    test_same_as_reference([6, 256], 2, 0);
}

#[test]
fn cat_should_match_reference_backend_dim1() {
    test_same_as_reference([6, 256], 2, 1);
}

#[test]
fn cat_should_support_uneven_launch() {
    test_same_as_reference([1, 137], 2, 0);
}

fn test_same_as_reference(shape: [usize; 2], num_tensors: usize, dim: usize) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);
    ref_device.seed(0);

    let tensors = (0..num_tensors)
        .map(|_| TestTensor::<2>::random(shape, Distribution::Default, &device))
        .collect::<Vec<_>>();
    let tensors_ref = tensors
        .iter()
        .map(|tensor| TestTensor::<2>::from_data(tensor.to_data(), &ref_device))
        .collect::<Vec<_>>();

    let tensor = TestTensor::<2>::cat(tensors, dim);
    let tensor_ref = TestTensor::<2>::cat(tensors_ref, dim);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&tensor_ref.into_data(), Tolerance::default());
}
