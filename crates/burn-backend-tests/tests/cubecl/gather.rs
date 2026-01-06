use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, Int, Shape, Tensor, backend::Backend};

#[test]
fn gather_should_work_with_multiple_workgroups_dim0() {
    test_same_as_ref([6, 256], 0);
}

#[test]
fn gather_should_work_with_multiple_workgroups_dim1() {
    test_same_as_ref([6, 256], 1);
}

fn test_same_as_ref<const D: usize>(shape: [usize; D], dim: usize) {
    let device = Default::default();
    TestBackend::seed(&device, 0);

    let max = shape[dim];
    let shape = Shape::new(shape);
    let tensor =
        Tensor::<TestBackend, D>::random(shape.clone(), Distribution::Default, &Default::default());
    let indices = Tensor::<TestBackend, 1, Int>::from_data(
        Tensor::<TestBackend, 1>::random(
            [shape.num_elements()],
            Distribution::Uniform(0., max as f64),
            &Default::default(),
        )
        .into_data(),
        &Default::default(),
    )
    .reshape(shape);
    let tensor_ref =
        Tensor::<ReferenceBackend, D>::from_data(tensor.to_data(), &Default::default());
    let indices_ref =
        Tensor::<ReferenceBackend, D, Int>::from_data(indices.to_data(), &Default::default());

    let actual = tensor.gather(dim, indices);
    let expected = tensor_ref.gather(dim, indices_ref);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}
