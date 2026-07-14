use super::*;
use burn_tensor::TensorData;

#[test]
fn test_tensor_full() {
    let device = Default::default();
    let int_tensor = TestTensorInt::<2>::full([2, 2], 2, &device);
    int_tensor
        .into_data()
        .assert_eq(&TensorData::from([[2, 2], [2, 2]]), false);
}
