use super::*;
use burn_tensor::TensorData;

#[test]
fn permute_bool() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(0..24, &device)
        .reshape([2, 3, 4])
        .greater_elem(10);

    let permuted = tensor.clone().permute([2, 1, 0]);

    // from pytorch:
    // import torch; torch.arange(0, 24).reshape(2, 3, 4).permute(2, 1, 0).gt(10)
    let expected = TensorData::from([
        [[false, true], [false, true], [false, true]],
        [[false, true], [false, true], [false, true]],
        [[false, true], [false, true], [false, true]],
        [[false, true], [false, true], [true, true]],
    ]);

    permuted.into_data().assert_eq(&expected, false);

    // Test with negative axis
    let permuted = tensor.clone().permute([-1, 1, 0]);
    permuted.into_data().assert_eq(&expected, false);

    // Test with the same axis
    let permuted = tensor.clone().permute([0, 1, 2]);
    permuted.into_data().assert_eq(&tensor.into_data(), false);
}
