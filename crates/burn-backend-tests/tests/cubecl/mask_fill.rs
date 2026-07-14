use super::*;
use burn_tensor::Distribution;
use burn_tensor::Tolerance;

#[test]
fn mask_fill_should_match_reference_backend() {
    let (tensor, mask, tensor_ref, mask_ref) = inputs_mask_fill();

    // MaskFillStrategy::Readonly
    let _clone_for_readonly = tensor.clone();

    let actual = tensor.mask_fill(mask, 4.0);
    let expected = tensor_ref.mask_fill(mask_ref, 4.0);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn mask_fill_inplace_should_match_reference_backend() {
    let (tensor, mask, tensor_ref, mask_ref) = inputs_mask_fill();

    // MaskFillStrategy::Inplace
    let actual = tensor.mask_fill(mask, 4.0);
    let expected = tensor_ref.mask_fill(mask_ref, 4.0);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[allow(clippy::type_complexity)]
fn inputs_mask_fill() -> (
    TestTensor<3>,
    TestTensorBool<3>,
    TestTensor<3>,
    TestTensorBool<3>,
) {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<3>::random([2, 6, 256], Distribution::Default, &device);
    let mask = TestTensor::<3>::random([2, 6, 256], Distribution::Uniform(0., 1.), &device)
        .lower_equal_elem(0.5);

    let tensor_ref = TestTensor::<3>::from_data(tensor.to_data(), &ref_device);
    let mask_ref = TestTensorBool::<3>::from_data(mask.to_data(), &ref_device);

    (tensor, mask, tensor_ref, mask_ref)
}
