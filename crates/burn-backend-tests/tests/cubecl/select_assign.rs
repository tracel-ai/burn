use super::*;
use burn_tensor::Distribution;
use burn_tensor::{Device, IndexingUpdateOp, Tolerance};

#[test]
fn select_add_should_work_with_multiple_workgroups_2d_dim0() {
    select_add_same_as_ref(0, [256, 6]);
}

#[test]
fn select_add_should_work_with_multiple_workgroups_2d_dim1() {
    select_add_same_as_ref(1, [6, 256]);
}

#[test]
fn select_add_should_work_with_multiple_workgroups_3d_dim0() {
    select_add_same_as_ref(0, [256, 6, 6]);
}

#[test]
fn select_add_should_work_with_multiple_workgroups_3d_dim1() {
    select_add_same_as_ref(1, [6, 256, 6]);
}

#[test]
fn select_add_should_work_with_multiple_workgroups_3d_dim2() {
    select_add_same_as_ref(2, [6, 6, 256]);
}

fn select_add_same_as_ref<const D: usize>(dim: usize, shape: [usize; D]) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let tensor = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let value = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let indices = TestTensorInt::<1>::random(
        [shape[dim]],
        Distribution::Uniform(0., shape[dim] as f64),
        &device,
    );
    let tensor_ref = TestTensor::<D>::from_data(tensor.to_data(), &ref_device);
    let value_ref = TestTensor::<D>::from_data(value.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<1>::from_data(indices.to_data(), &ref_device);

    let actual = tensor.select_assign(dim, indices, value, IndexingUpdateOp::Add);
    let expected = tensor_ref.select_assign(dim, indices_ref, value_ref, IndexingUpdateOp::Add);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}
