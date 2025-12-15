use super::qtensor::*;
use super::*;
use alloc::vec;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn test_split_evenly_divisible() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    let tensors = tensor.split(2, 0);
    assert_eq!(tensors.len(), 3);

    let expected = [
        TensorData::from([0., 1.]),
        TensorData::from([2., 3.]),
        TensorData::from([4., 5.]),
    ];

    for (index, tensor) in tensors.into_iter().enumerate() {
        tensor
            .dequantize()
            .to_data()
            .assert_approx_eq::<FloatElem>(&expected[index], Tolerance::absolute(1e-1));
    }
}

#[test]
fn test_split_not_evenly_divisible() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let tensors = tensor.split(2, 0);
    assert_eq!(tensors.len(), 4);

    let expected = [
        TensorData::from([0., 1.]),
        TensorData::from([2., 3.]),
        TensorData::from([4., 5.]),
        TensorData::from([6.]),
    ];

    for (index, tensor) in tensors.into_iter().enumerate() {
        tensor
            .dequantize()
            .to_data()
            .assert_approx_eq::<FloatElem>(&expected[index], Tolerance::absolute(1e-1));
    }
}

#[test]
fn test_split_along_dim1() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let tensors = tensor.split(2, 1);
    assert_eq!(tensors.len(), 2);

    let expected = [
        TensorData::from([[0., 1.], [3., 4.]]),
        TensorData::from([[2.], [5.]]),
    ];

    for (index, tensor) in tensors.into_iter().enumerate() {
        tensor
            .dequantize()
            .to_data()
            .assert_approx_eq::<FloatElem>(&expected[index], Tolerance::absolute(1e-1));
    }
}

#[test]
fn test_split_split_size_larger_than_tensor_size() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    let tensors = tensor.split(10, 0);
    assert_eq!(tensors.len(), 1);

    let expected = [TensorData::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])];

    for (index, tensor) in tensors.into_iter().enumerate() {
        tensor
            .dequantize()
            .to_data()
            .assert_approx_eq::<FloatElem>(&expected[index], Tolerance::absolute(1e-1));
    }
}

#[test]
#[should_panic(
    expected = "split_size must be greater than 0 unless the tensor size along the dimension is 0."
)]
fn test_split_with_zero_split_size_non_zero_tensor() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    let _ = tensor.split(0, 0);
}

#[test]
#[should_panic(expected = "Given dimension is greater than or equal to the tensor rank.")]
fn test_split_invalid_dim() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    let _ = tensor.split(1, 2);
}

#[test]
fn test_split_with_sizes() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    let tensors = tensor.split_with_sizes(vec![2, 3, 1], 0);
    assert_eq!(tensors.len(), 3);

    let expected = [
        TensorData::from([0., 1.]),
        TensorData::from([2., 3., 4.]),
        TensorData::from([5.]),
    ];

    for (index, tensor) in tensors.into_iter().enumerate() {
        tensor
            .dequantize()
            .to_data()
            .assert_approx_eq::<FloatElem>(&expected[index], Tolerance::absolute(1e-1));
    }
}

#[test]
#[should_panic(
    expected = "The sum of split_sizes must equal the tensor size along the specified dimension."
)]
fn test_split_with_sizes_invalid_sum() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    let _ = tensor.split_with_sizes(vec![2, 2, 1], 0);
}

#[test]
fn test_split_with_sizes_zero_length() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 2.0, 5.0]);

    let tensors = tensor.split_with_sizes(vec![0, 1, 2], 0);
    assert_eq!(tensors.len(), 2);

    let expected = [TensorData::from([0.]), TensorData::from([2., 5.])];

    for (index, tensor) in tensors.into_iter().enumerate() {
        tensor
            .dequantize()
            .to_data()
            .assert_approx_eq::<FloatElem>(&expected[index], Tolerance::absolute(1e-1));
    }
}
