use super::*;
use burn_tensor::Distribution;
use burn_tensor::s;

#[test]
fn test_unfold_float() {
    let device = Default::default();

    let input = TestTensor::<3>::random([2, 6, 6], Distribution::Default, &device);

    let dim = 1;
    let size = 3;
    let step = 2;
    let actual: TestTensor<4> = input.clone().unfold(dim, size, step);

    let expected = TestTensor::<4>::empty([2, 2, 6, 3], &device)
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
    if (IntElem::MAX as u32) < 255 - 1 {
        return;
    }
    let device = Default::default();

    let input = TestTensorInt::<3>::random([2, 6, 6], Distribution::Default, &device);

    let dim = 1;
    let size = 3;
    let step = 2;
    let actual: TestTensorInt<4> = input.clone().unfold(dim, size, step);

    let expected = TestTensorInt::<4>::empty([2, 2, 6, 3], &device)
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

    let input =
        TestTensor::<3>::random([2, 6, 6], Distribution::Default, &device).greater_elem(0.5);

    let dim = 1;
    let size = 3;
    let step = 2;
    let actual: TestTensorBool<4> = input.clone().unfold(dim, size, step);

    let expected = TestTensorBool::<4>::empty([2, 2, 6, 3], &device)
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
