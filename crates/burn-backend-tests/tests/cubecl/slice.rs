use super::*;
use burn_tensor::Distribution;
use burn_tensor::Tolerance;

#[test]
fn slice_should_work_with_multiple_workgroups() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<2>::random([6, 256], Distribution::Default, &device);
    let indices = [3..5, 45..256];
    let tensor_ref = TestTensor::<2>::from_data(tensor.to_data(), &ref_device);

    let actual = tensor.slice(indices.clone());
    let expected = tensor_ref.slice(indices);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn slice_with_32b_aligned_end_remains_a_view() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<3>::random([4, 32, 8], Distribution::Default, &device);
    let tensor_ref = TestTensor::<3>::from_data(tensor.to_data(), &ref_device);
    let ranges = [0..4, 0..7, 0..8];

    let actual = tensor.clone().slice(ranges.clone());
    let expected = tensor_ref.slice(ranges);

    #[cfg(not(feature = "fusion"))]
    assert!(!actual.can_mut() && !tensor.can_mut());

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}
