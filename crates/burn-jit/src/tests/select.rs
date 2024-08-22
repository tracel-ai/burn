#[burn_tensor_testgen::testgen(select)]
mod tests {
    use super::*;
    use burn_tensor::{Distribution, Int, Tensor};

    #[test]
    fn select_should_work_with_multiple_workgroups() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let indices = Tensor::<TestBackend, 1, Int>::arange(0..100, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let indices_ref =
            Tensor::<ReferenceBackend, 1, Int>::from_data(indices.to_data(), &Default::default());

        let actual = tensor.select(1, indices);
        let expected = tensor_ref.select(1, indices_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
}
