use super::*;
use burn_tensor::TensorData;

#[test]
fn test_chunk_evenly_divisible() {
    let tensors = TestTensorInt::arange(0..12, &Default::default()).chunk(6, 0);
    assert_eq!(tensors.len(), 6);

    let expected = vec![
        TensorData::from([0, 1]),
        TensorData::from([2, 3]),
        TensorData::from([4, 5]),
        TensorData::from([6, 7]),
        TensorData::from([8, 9]),
        TensorData::from([10, 11]),
    ];

    for (index, tensor) in tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
fn test_chunk_not_evenly_divisible() {
    let tensors = TestTensorInt::arange(0..11, &Default::default()).chunk(6, 0);
    assert_eq!(tensors.len(), 6);

    let expected = vec![
        TensorData::from([0, 1]),
        TensorData::from([2, 3]),
        TensorData::from([4, 5]),
        TensorData::from([6, 7]),
        TensorData::from([8, 9]),
        TensorData::from([10]),
    ];

    for (index, tensor) in tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
fn test_chunk_not_evenly_divisible_remains_several() {
    let tensors = TestTensorInt::arange(0..100, &Default::default()).chunk(8, 0);
    assert_eq!(tensors.len(), 8);

    let expected = [13, 13, 13, 13, 13, 13, 13, 9];

    for (index, tensor) in tensors.iter().enumerate() {
        assert_eq!(tensor.shape().dims[0], expected[index]);
    }
}

#[test]
fn test_chunk_not_divisible() {
    let tensors = TestTensorInt::arange(0..6, &Default::default()).chunk(7, 0);
    assert_eq!(tensors.len(), 6);

    let expected = vec![
        TensorData::from([0]),
        TensorData::from([1]),
        TensorData::from([2]),
        TensorData::from([3]),
        TensorData::from([4]),
        TensorData::from([5]),
    ];

    for (index, tensor) in tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
fn test_chunk_multi_dimension() {
    let tensors =
        TestTensorInt::<2>::from_data(TensorData::from([[0, 1, 2, 3]]), &Default::default())
            .chunk(2, 1);
    assert_eq!(tensors.len(), 2);

    let expected = vec![TensorData::from([[0, 1]]), TensorData::from([[2, 3]])];

    for (index, tensor) in tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
#[should_panic]
fn test_invalid_dim() {
    let _tensors = TestTensorInt::arange(0..12, &Default::default()).chunk(6, 1);
}
