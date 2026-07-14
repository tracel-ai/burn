use super::*;
use burn_tensor::TensorData;

#[test]
fn test_chunk_multi_dimension() {
    let tensors =
        TestTensorInt::<2>::from_data(TensorData::from([[0, 1, 2, 3]]), &Default::default())
            .chunk(2, 1);
    assert_eq!(tensors.len(), 2);

    let expected = [TensorData::from([[0, 1]]), TensorData::from([[2, 3]])];

    for (index, tensor) in tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}
