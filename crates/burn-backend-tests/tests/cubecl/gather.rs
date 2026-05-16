use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Device, Distribution, Shape};

#[test]
fn gather_should_work_with_multiple_workgroups_dim0() {
    test_same_as_ref([6, 256], 0);
}

#[test]
fn gather_should_work_with_multiple_workgroups_dim1() {
    test_same_as_ref([6, 256], 1);
}

fn test_same_as_ref<const D: usize>(shape: [usize; D], dim: usize) {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let max = shape[dim];
    let shape = Shape::new(shape);
    let tensor = TestTensor::<D>::random(shape.clone(), Distribution::Default, &device);
    let indices = TestTensorInt::<1>::from_data(
        TestTensor::<1>::random(
            [shape.num_elements()],
            Distribution::Uniform(0., max as f64),
            &device,
        )
        .into_data(),
        &device,
    )
    .reshape(shape);
    let tensor_ref = TestTensor::<D>::from_data(tensor.to_data(), &ref_device);
    let indices_ref = TestTensorInt::<D>::from_data(indices.to_data(), &ref_device);

    let actual = tensor.gather(dim, indices);
    let expected = tensor_ref.gather(dim, indices_ref);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}
