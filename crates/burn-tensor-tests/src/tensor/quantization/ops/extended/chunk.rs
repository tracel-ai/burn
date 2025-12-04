use crate::qtensor::*;
use crate::*;
use alloc::vec::Vec;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn test_chunk_evenly_divisible() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    let tensors: Vec<TestTensor<1>> = tensor.chunk(3, 0);
    assert_eq!(tensors.len(), 3);

    let expected = vec![
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
fn test_chunk_not_evenly_divisible() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let tensors: Vec<TestTensor<1>> = tensor.chunk(4, 0);
    assert_eq!(tensors.len(), 4);

    let expected = vec![
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
fn test_chunk_not_divisible() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    let tensors: Vec<TestTensor<1>> = tensor.chunk(7, 0);
    assert_eq!(tensors.len(), 6);

    let expected = vec![
        TensorData::from([0.]),
        TensorData::from([1.]),
        TensorData::from([2.]),
        TensorData::from([3.]),
        TensorData::from([4.]),
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
fn test_chunk_multi_dimension() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]);

    let tensors: Vec<TestTensor<2>> = tensor.chunk(2, 1);
    assert_eq!(tensors.len(), 2);

    let expected = vec![
        TensorData::from([[0., 1., 2.]]),
        TensorData::from([[3., 4., 5.]]),
    ];

    
    for (index, tensor) in tensors.into_iter().enumerate() {
        tensor
            .dequantize()
            .to_data()
            .assert_approx_eq::<FloatElem>(&expected[index], Tolerance::absolute(1e-1));
    }
}

#[test]
#[should_panic]
fn test_invalid_dim() {
    let _tensors = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).chunk(6, 1);
}
