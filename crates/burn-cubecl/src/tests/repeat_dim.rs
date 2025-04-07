#[burn_tensor_testgen::testgen(repeat_dim)]
mod tests {
    use super::*;
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn repeat_dim_0_few_times() {
        let tensor =
            Tensor::<TestBackend, 3>::random([1, 6, 6], Distribution::Default, &Default::default());
        let dim = 0;
        let times = 4;
        let tensor_ref =
            Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &Default::default());

        let actual = tensor.repeat_dim(dim, times);
        let expected = tensor_ref.repeat_dim(dim, times);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    #[test]
    fn repeat_dim_1_few_times() {
        let tensor =
            Tensor::<TestBackend, 3>::random([6, 1, 6], Distribution::Default, &Default::default());
        let dim = 1;
        let times = 4;
        let tensor_ref =
            Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &Default::default());

        let actual = tensor.repeat_dim(dim, times);
        let expected = tensor_ref.repeat_dim(dim, times);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    #[test]
    fn repeat_dim_2_few_times() {
        let tensor =
            Tensor::<TestBackend, 3>::random([6, 6, 1], Distribution::Default, &Default::default());
        let dim = 2;
        let times = 4;
        let tensor_ref =
            Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &Default::default());

        let actual = tensor.repeat_dim(dim, times);
        let expected = tensor_ref.repeat_dim(dim, times);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    #[test]
    fn repeat_dim_2_many_times() {
        let tensor = Tensor::<TestBackend, 3>::random(
            [10, 10, 1],
            Distribution::Default,
            &Default::default(),
        );
        let dim = 2;
        let times = 200;
        let tensor_ref =
            Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &Default::default());

        let actual = tensor.repeat_dim(dim, times);
        let expected = tensor_ref.repeat_dim(dim, times);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
}
