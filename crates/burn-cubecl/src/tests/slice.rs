#[burn_tensor_testgen::testgen(slice)]
mod tests {
    use super::*;
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn slice_should_work_with_multiple_workgroups() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let indices = [3..5, 45..256];
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let actual = tensor.slice(indices.clone());
        let expected = tensor_ref.slice(indices);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
}
