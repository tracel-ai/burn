use super::*;
use burn_tensor::TensorData;

#[test]
fn movedim_bool() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(0..24, &device)
        .reshape([2, 3, 4])
        .greater_elem(10);

    let permuted = tensor.clone().movedim(0, 2);
    // from pytorch:
    // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim(0, 2).gt(10)
    let expected = TensorData::from([
        [[false, true], [false, true], [false, true], [false, true]],
        [[false, true], [false, true], [false, true], [false, true]],
        [[false, true], [false, true], [false, true], [true, true]],
    ]);

    permuted.into_data().assert_eq(&expected, false);

    // Test with negative axis
    let permuted = tensor.clone().movedim(0, -1);
    permuted.into_data().assert_eq(&expected, false);

    // Test with the same axis
    let permuted = tensor.clone().movedim(0, 0);
    permuted.into_data().assert_eq(&tensor.into_data(), false);
}

#[test]
fn vec_input_bool() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(0..24, &device)
        .reshape([2, 3, 4])
        .greater_elem(10);

    let permuted = tensor.clone().movedim(vec![0, 1], vec![1, 0]);
    // from pytorch
    // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim([0, 1], [1, 0]).gt(10)
    let expected = TensorData::from([
        [[false, false, false, false], [true, true, true, true]],
        [[false, false, false, false], [true, true, true, true]],
        [[false, false, false, true], [true, true, true, true]],
    ]);

    permuted.into_data().assert_eq(&expected, false);

    // Test with negative axes
    let permuted = tensor.clone().movedim(vec![-3, -2], vec![-2, -3]);
    permuted.into_data().assert_eq(&expected, false);

    // Test with the same axes
    let permuted = tensor.clone().movedim(vec![0, 1], vec![0, 1]);
    permuted.into_data().assert_eq(&tensor.into_data(), false);
}
