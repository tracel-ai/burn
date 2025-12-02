#[burn_tensor_testgen::testgen(select_assign)]
mod tests {
    use super::*;
    use burn_tensor::{Distribution, Int, Tensor, backend::Backend};
    use burn_tensor::{IndexingUpdateOp, Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

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
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let tensor =
            Tensor::<TestBackend, D>::random(shape, Distribution::Default, &Default::default());
        let value =
            Tensor::<TestBackend, D>::random(shape, Distribution::Default, &Default::default());
        let indices = Tensor::<TestBackend, 1, Int>::random(
            [shape[dim]],
            Distribution::Uniform(0., shape[dim] as f64),
            &Default::default(),
        );
        let tensor_ref =
            Tensor::<ReferenceBackend, D>::from_data(tensor.to_data(), &Default::default());
        let value_ref =
            Tensor::<ReferenceBackend, D>::from_data(value.to_data(), &Default::default());
        let indices_ref =
            Tensor::<ReferenceBackend, 1, Int>::from_data(indices.to_data(), &Default::default());

        let actual = tensor.select_assign(dim, indices, value, IndexingUpdateOp::Add);
        let expected = tensor_ref.select_assign(dim, indices_ref, value_ref, IndexingUpdateOp::Add);

        expected
            .into_data()
            .assert_approx_eq::<FT>(&actual.into_data(), Tolerance::default());
    }
}
