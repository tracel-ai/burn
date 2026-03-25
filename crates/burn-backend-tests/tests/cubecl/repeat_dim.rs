use super::*;
use burn_tensor::Distribution;
use burn_tensor::Tolerance;

#[test]
fn repeat_dim_0_few_times() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<3>::random([1, 6, 6], Distribution::Default, &device);
    let dim = 0;
    let times = 4;
    let tensor_ref = TestTensor::<3>::from_data(tensor.to_data(), &ref_device);

    let actual = tensor.repeat_dim(dim, times);
    let expected = tensor_ref.repeat_dim(dim, times);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn repeat_dim_1_few_times() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<3>::random([6, 1, 6], Distribution::Default, &device);
    let dim = 1;
    let times = 4;
    let tensor_ref = TestTensor::<3>::from_data(tensor.to_data(), &ref_device);

    let actual = tensor.repeat_dim(dim, times);
    let expected = tensor_ref.repeat_dim(dim, times);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn repeat_dim_2_few_times() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<3>::random([6, 6, 1], Distribution::Default, &device);
    let dim = 2;
    let times = 4;
    let tensor_ref = TestTensor::<3>::from_data(tensor.to_data(), &ref_device);

    let actual = tensor.repeat_dim(dim, times);
    let expected = tensor_ref.repeat_dim(dim, times);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn repeat_dim_2_many_times() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<3>::random([10, 10, 1], Distribution::Default, &device);
    let dim = 2;
    let times = 200;
    let tensor_ref = TestTensor::<3>::from_data(tensor.to_data(), &ref_device);

    let actual = tensor.repeat_dim(dim, times);
    let expected = tensor_ref.repeat_dim(dim, times);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}
