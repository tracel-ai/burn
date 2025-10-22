#[burn_tensor_testgen::testgen(unfold)]
mod tests {
    use super::*;
    use burn_tensor::s;
    use burn_tensor::{Bool, Int};
    use burn_tensor::{Distribution, Tensor, backend::Backend, module};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_unfold_float() {
        let device = Default::default();

        let input = Tensor::<TestBackend, 3>::random([2, 6, 6], Distribution::Default, &device);

        let dim = 1;
        let size = 3;
        let step = 2;
        let actual: Tensor<TestBackend, 4> = input.clone().unfold(dim, size, step);

        let expected = Tensor::<TestBackend, 4>::empty([2, 2, 6, 3], &device)
            .slice_assign(
                s![.., 0, .., ..],
                input
                    .clone()
                    .slice(s![.., 0..3, ..])
                    .swap_dims(1, 2)
                    .unsqueeze_dim::<4>(1),
            )
            .slice_assign(
                s![.., 1, .., ..],
                input
                    .clone()
                    .slice(s![.., 2..5, ..])
                    .swap_dims(1, 2)
                    .unsqueeze_dim::<4>(1),
            );

        actual.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_unfold_int() {
        // Distribution::Default samples from [0, 255)
        if (IntType::MAX as u32) < 255 - 1 {
            return;
        }

        let device = Default::default();

        let input =
            Tensor::<TestBackend, 3, Int>::random([2, 6, 6], Distribution::Default, &device);

        let dim = 1;
        let size = 3;
        let step = 2;
        let actual: Tensor<TestBackend, 4, Int> = input.clone().unfold(dim, size, step);

        let expected = Tensor::<TestBackend, 4, Int>::empty([2, 2, 6, 3], &device)
            .slice_assign(
                s![.., 0, .., ..],
                input
                    .clone()
                    .slice(s![.., 0..3, ..])
                    .swap_dims(1, 2)
                    .unsqueeze_dim::<4>(1),
            )
            .slice_assign(
                s![.., 1, .., ..],
                input
                    .clone()
                    .slice(s![.., 2..5, ..])
                    .swap_dims(1, 2)
                    .unsqueeze_dim::<4>(1),
            );

        actual.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_unfold_bool() {
        let device = Default::default();

        let input: Tensor<TestBackend, 3, Bool> =
            Tensor::<TestBackend, 3>::random([2, 6, 6], Distribution::Default, &device)
                .greater_elem(0.5);

        let dim = 1;
        let size = 3;
        let step = 2;
        let actual: Tensor<TestBackend, 4, Bool> = input.clone().unfold(dim, size, step);

        let expected = Tensor::<TestBackend, 4, Bool>::empty([2, 2, 6, 3], &device)
            .slice_assign(
                s![.., 0, .., ..],
                input
                    .clone()
                    .slice(s![.., 0..3, ..])
                    .swap_dims(1, 2)
                    .unsqueeze_dim::<4>(1),
            )
            .slice_assign(
                s![.., 1, .., ..],
                input
                    .clone()
                    .slice(s![.., 2..5, ..])
                    .swap_dims(1, 2)
                    .unsqueeze_dim::<4>(1),
            );

        actual.to_data().assert_eq(&expected.to_data(), true);
    }
}
