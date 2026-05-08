use super::*;
use burn_tensor::Device;
use burn_tensor::Distribution;
use burn_tensor::Tolerance;

#[test]
fn mask_where_should_match_reference_backend() {
    let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

    let actual = tensor.mask_where(mask, value);
    let expected = tensor_ref.mask_where(mask_ref, value_ref);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}
#[test]
fn mask_where_inplace_lhs_should_match_reference_backend() {
    let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

    // MaskWhereStrategy::InplaceLhs
    let actual = tensor.mask_where(mask, value);
    let expected = tensor_ref.mask_where(mask_ref, value_ref);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn mask_where_inplace_rhs_should_match_reference_backend() {
    let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

    // MaskWhereStrategy::InplaceRhs
    let _clone_for_inplace_rhs = tensor.clone();

    let actual = tensor.mask_where(mask, value);
    let expected = tensor_ref.mask_where(mask_ref, value_ref);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[allow(clippy::type_complexity)]
fn inputs_mask_where() -> (
    TestTensor<3>,
    TestTensor<3>,
    TestTensorBool<3>,
    TestTensor<3>,
    TestTensor<3>,
    TestTensorBool<3>,
) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let tensor = TestTensor::<3>::random([2, 6, 256], Distribution::Default, &device);
    let value = TestTensor::<3>::random([2, 6, 256], Distribution::Default, &device);
    let mask = TestTensor::<3>::random([2, 6, 256], Distribution::Uniform(0., 1.), &device)
        .lower_equal_elem(0.5);

    let tensor_ref = TestTensor::<3>::from_data(tensor.to_data(), &ref_device);
    let value_ref = TestTensor::<3>::from_data(value.to_data(), &ref_device);
    let mask_ref = TestTensorBool::<3>::from_data(mask.to_data(), &ref_device);
    mask.to_data().assert_eq(&mask_ref.to_data(), false);

    (tensor, value, mask, tensor_ref, value_ref, mask_ref)
}
