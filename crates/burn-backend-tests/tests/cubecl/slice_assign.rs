use super::*;
use burn_tensor::{Distribution, Tolerance};

#[test]
fn slice_assign_should_work_with_multiple_workgroups() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<2>::random([6, 256], Distribution::Default, &device);
    let value = TestTensor::<2>::random([2, 211], Distribution::Default, &device);
    let indices = [3..5, 45..256];
    let tensor_ref = TestTensor::<2>::from_data(tensor.to_data(), &ref_device);
    let value_ref = TestTensor::<2>::from_data(value.to_data(), &ref_device);

    let actual = tensor.slice_assign(indices.clone(), value);
    let expected = tensor_ref.slice_assign(indices, value_ref);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}
