use super::*;
use burn_tensor::TensorData;

#[test]
fn permute_float_a() {
    let tensor = TestTensor::<1>::from([
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
    ])
    .reshape([2, 2, 4]);

    let permuted = tensor.clone().permute([2, 1, 0]);

    let expected = TensorData::from([
        [[0., 8.], [4., 12.]],
        [[1., 9.], [5., 13.]],
        [[2., 10.], [6., 14.]],
        [[3., 11.], [7., 15.]],
    ]);

    permuted.into_data().assert_eq(&expected, false);

    // Test with negative axis
    let permuted = tensor.clone().permute([-1, 1, 0]);
    permuted.into_data().assert_eq(&expected, false);

    // Test with the same axis
    let permuted = tensor.clone().permute([0, 1, 2]);
    permuted.into_data().assert_eq(&tensor.into_data(), false);
}

#[test]
fn permute_float() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(0..24, &device)
        .reshape([2, 3, 4])
        .float();

    let permuted = tensor.clone().permute([2, 1, 0]);

    // from pytorch:
    // import torch; torch.arange(0, 24).reshape(2, 3, 4).permute(2, 1, 0).float()
    let expected = TensorData::from([
        [[0., 12.], [4., 16.], [8., 20.]],
        [[1., 13.], [5., 17.], [9., 21.]],
        [[2., 14.], [6., 18.], [10., 22.]],
        [[3., 15.], [7., 19.], [11., 23.]],
    ]);

    permuted.into_data().assert_eq(&expected, false);

    // Test with negative axis
    let permuted = tensor.clone().permute([-1, 1, 0]);
    permuted.into_data().assert_eq(&expected, false);

    // Test with the same axis
    let permuted = tensor.clone().permute([0, 1, 2]);
    permuted.into_data().assert_eq(&tensor.into_data(), true);
}
