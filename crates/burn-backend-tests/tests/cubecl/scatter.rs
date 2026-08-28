use super::*;
use burn_tensor::Distribution;
use burn_tensor::{Device, IndexingUpdateOp, Tolerance};

#[test]
fn scatter_should_work_with_multiple_workgroups_2d_dim0() {
    same_as_reference_same_shape(0, [256, 32]);
}

#[test]
fn scatter_should_work_with_multiple_workgroups_2d_dim1() {
    same_as_reference_same_shape(1, [32, 256]);
}

#[test]
fn scatter_should_work_with_multiple_workgroups_3d_dim0() {
    same_as_reference_same_shape(0, [256, 6, 6]);
}

#[test]
fn scatter_should_work_with_multiple_workgroups_3d_dim1() {
    same_as_reference_same_shape(1, [6, 256, 6]);
}

#[test]
fn scatter_should_work_with_multiple_workgroups_3d_dim2() {
    same_as_reference_same_shape(2, [6, 6, 256]);
}

#[test]
fn scatter_should_work_with_multiple_workgroups_diff_shapes() {
    same_as_reference_diff_shape(1, [32, 128], [32, 1]);
}

#[test]
fn scatter_mul_should_work_with_multiple_workgroups_2d_dim0() {
    mul_same_as_reference(0, [256, 32]);
}

#[test]
fn scatter_mul_should_work_with_multiple_workgroups_2d_dim1() {
    mul_same_as_reference(1, [32, 256]);
}

fn same_as_reference_diff_shape<const D: usize>(
    dim: usize,
    shape1: [usize; D],
    shape2: [usize; D],
) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let tensor = TestTensor::<D>::random(shape1, Distribution::Default, &device);
    let value = TestTensor::<D>::random(shape2, Distribution::Default, &device);
    let indices = TestTensorInt::<1>::random(
        [shape2.iter().product::<usize>()],
        Distribution::Uniform(0., shape2[dim] as f64),
        &device,
    )
    .reshape(shape2);

    let tensor_ref = TestTensor::<D>::from_data(tensor.to_data(), &ref_device);
    let value_ref = TestTensor::<D>::from_data(value.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<D>::from_data(indices.to_data(), &ref_device);

    let actual = tensor.scatter(dim, indices, value, IndexingUpdateOp::Add);
    let expected = tensor_ref.scatter(dim, indices_ref, value_ref, IndexingUpdateOp::Add);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

fn same_as_reference_same_shape<const D: usize>(dim: usize, shape: [usize; D]) {
    same_as_reference_diff_shape(dim, shape, shape);
}

fn mul_same_as_reference<const D: usize>(dim: usize, shape: [usize; D]) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let tensor = TestTensor::<D>::random(shape, Distribution::Default, &device);
    let value = TestTensor::<D>::random(shape, Distribution::Default, &device);

    // Keep indices unique along the updated dimension because duplicate-index behavior for
    // multiplicative updates is intentionally unspecified.
    let mut index_shape = [1; D];
    index_shape[dim] = shape[dim];
    let mut indices: TestTensorInt<D> =
        TestTensorInt::<1>::arange(0..shape[dim] as i64, &device).reshape(index_shape);
    for axis in 0..D {
        if axis != dim {
            indices = indices.repeat_dim(axis, shape[axis]);
        }
    }

    let tensor_ref = TestTensor::<D>::from_data(tensor.to_data(), &ref_device);
    let value_ref = TestTensor::<D>::from_data(value.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<D>::from_data(indices.to_data(), &ref_device);

    let actual = tensor.scatter(dim, indices, value, IndexingUpdateOp::Mul);
    let expected = tensor_ref.scatter(dim, indices_ref, value_ref, IndexingUpdateOp::Mul);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}
