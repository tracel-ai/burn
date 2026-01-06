use super::*;
use burn_tensor::TensorData;

#[test]
fn flip_int() {
    let device = Default::default();
    let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

    let flipped = tensor.clone().flip([0, 2]);
    // from pytorch:
    // import torch; torch.arange(0, 24).reshape(2, 3, 4).flip((0, 2))
    let expected = TensorData::from([
        [[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]],
        [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]],
    ]);

    flipped.into_data().assert_eq(&expected, false);

    // Test with no flip
    let flipped = tensor.clone().flip([]);
    assert_eq!(tensor.into_data(), flipped.into_data());
}
