use super::*;
use burn_tensor::TensorData;

#[test]
fn flip_bool() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(0..24, &device)
        .reshape([2, 3, 4])
        .greater_elem(10);

    let flipped = tensor.clone().flip([0, 2]);

    // from pytorch:
    // import torch; torch.arange(0, 24).reshape(2, 3, 4).flip((0, 2)).gt(10)
    let data_expected = TensorData::from([
        [
            [true, true, true, true],
            [true, true, true, true],
            [true, true, true, true],
        ],
        [
            [false, false, false, false],
            [false, false, false, false],
            [true, false, false, false],
        ],
    ]);

    flipped.into_data().assert_eq(&data_expected, false);

    // Test with no flip
    let flipped = tensor.clone().flip([]);
    tensor.into_data().assert_eq(&flipped.into_data(), false);
}
