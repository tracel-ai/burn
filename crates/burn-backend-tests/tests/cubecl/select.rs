use super::*;
use burn_tensor::Distribution;
use burn_tensor::Tolerance;

#[test]
fn select_should_work_with_multiple_workgroups() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<2>::random([6, 256], Distribution::Default, &device);
    let indices = TestTensorInt::<1>::arange(0..100, &device);
    let tensor_ref = TestTensor::<2>::from_data(tensor.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<1>::from_data(indices.to_data(), &ref_device);

    let actual = tensor.select(1, indices);
    let expected = tensor_ref.select(1, indices_ref);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}
