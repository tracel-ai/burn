use super::*;
use burn_tensor::Distribution;
use burn_tensor::TensorData;
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
fn test_unfold_reshape_overlap() {
    let device = Default::default();
    let signal = TestTensor::<2>::from_data(
        [[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]],
        &device,
    );

    let size = 4;
    let step = 1;

    let frames = signal.unfold::<3, _>(1, size, step);

    let flat = frames.reshape([9, 4]);

    let data = flat.into_data();

    let expected = TensorData::from([
        [1., 2., 3., 4.],
        [2., 3., 4., 5.],
        [3., 4., 5., 6.],
        [4., 5., 6., 7.],
        [5., 6., 7., 8.],
        [6., 7., 8., 9.],
        [7., 8., 9., 10.],
        [8., 9., 10., 11.],
        [9., 10., 11., 12.],
    ]);

    data.assert_eq(&expected, false);
}
