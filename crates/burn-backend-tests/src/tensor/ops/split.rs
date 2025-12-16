use super::*;
use burn_tensor::TensorData;

#[test]
fn test_split_evenly_divisible() {
    let device = Default::default();
    let tensors =
        TestTensor::<2>::from_data([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], &device);

    let split_tensors = tensors.split(2, 0);
    assert_eq!(split_tensors.len(), 3);

    let expected = [
        TensorData::from([[0, 1], [2, 3]]),
        TensorData::from([[4, 5], [6, 7]]),
        TensorData::from([[8, 9], [10, 11]]),
    ];

    for (index, tensor) in split_tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
fn test_split_not_evenly_divisible() {
    let device = Default::default();
    let tensors = TestTensor::<2>::from_data([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], &device);

    let split_tensors = tensors.split(2, 0);
    assert_eq!(split_tensors.len(), 3);

    let expected = [
        TensorData::from([[0, 1], [2, 3]]),
        TensorData::from([[4, 5], [6, 7]]),
        TensorData::from([[8, 9]]),
    ];

    for (index, tensor) in split_tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
fn test_split_along_dim1() {
    let device = Default::default();
    let tensors = TestTensor::<2>::from_data([[0, 1, 2], [3, 4, 5]], &device);

    let split_tensors = tensors.split(2, 1);
    assert_eq!(split_tensors.len(), 2);

    let expected = [
        TensorData::from([[0, 1], [3, 4]]),
        TensorData::from([[2], [5]]),
    ];

    for (index, tensor) in split_tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
fn test_split_split_size_larger_than_tensor_size() {
    let device = Default::default();
    let tensors = TestTensor::<1>::from_data([0, 1, 2, 3, 4], &device);

    let split_tensors = tensors.split(10, 0);
    assert_eq!(split_tensors.len(), 1);

    let expected = [TensorData::from([0, 1, 2, 3, 4])];

    for (index, tensor) in split_tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
fn test_split_with_zero_split_size_zero_tensor_size() {
    let device = Default::default();
    let empty_array: [i32; 0] = [];
    let tensors = TestTensor::<1>::from_data(empty_array, &device);

    let split_tensors = tensors.split(0, 0);
    assert_eq!(split_tensors.len(), 0);
}

#[test]
fn test_split_zero_sized_tensor() {
    let device = Default::default();
    let empty_array: [i32; 0] = [];
    let tensors = TestTensor::<1>::from_data(empty_array, &device);

    let split_tensors = tensors.split(1, 0);
    assert_eq!(split_tensors.len(), 0);
}

#[test]
#[should_panic(
    expected = "split_size must be greater than 0 unless the tensor size along the dimension is 0."
)]
fn test_split_with_zero_split_size_non_zero_tensor() {
    let device = Default::default();
    let tensors = TestTensor::<1>::from_data([0, 1, 2, 3, 4], &device);

    let _split_tensors = tensors.split(0, 0);
}

#[test]
#[should_panic(expected = "Given dimension is greater than or equal to the tensor rank.")]
fn test_split_invalid_dim() {
    let device = Default::default();
    let tensors = TestTensor::<1>::from_data([0, 1, 2], &device);

    let _split_tensors = tensors.split(1, 2);
}

#[test]
fn test_split_3d_tensor_along_dim0() {
    let device = Default::default();
    let tensors = TestTensor::<3>::from_data(
        [
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
            [[8, 9], [10, 11]],
            [[12, 13], [14, 15]],
        ],
        &device,
    );

    let split_tensors = tensors.split(2, 0);
    assert_eq!(split_tensors.len(), 2);

    let expected = [
        TensorData::from([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
        TensorData::from([[[8, 9], [10, 11]], [[12, 13], [14, 15]]]),
    ];

    for (index, tensor) in split_tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
fn test_split_3d_tensor_along_dim1() {
    let device = Default::default();
    let tensors = TestTensor::<3>::from_data(
        [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]],
        &device,
    );

    let split_tensors = tensors.split(2, 1);
    assert_eq!(split_tensors.len(), 2);

    let expected = [
        TensorData::from([[[0, 1], [2, 3]], [[6, 7], [8, 9]]]),
        TensorData::from([[[4, 5]], [[10, 11]]]),
    ];

    for (index, tensor) in split_tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
fn test_split_with_sizes() {
    let device = Default::default();
    let tensors = TestTensor::<1>::from_data([0, 1, 2, 3, 4, 5], &device);

    let split_tensors = tensors.split_with_sizes(vec![2, 3, 1], 0);
    assert_eq!(split_tensors.len(), 3);

    let expected = [
        TensorData::from([0, 1]),
        TensorData::from([2, 3, 4]),
        TensorData::from([5]),
    ];

    for (index, tensor) in split_tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}

#[test]
#[should_panic(
    expected = "The sum of split_sizes must equal the tensor size along the specified dimension."
)]
fn test_split_with_sizes_invalid_sum() {
    let device = Default::default();
    let tensors = TestTensor::<1>::from_data([0, 1, 2, 3, 4, 5], &device);

    let _split_tensors = tensors.split_with_sizes(vec![2, 2, 1], 0);
}

#[test]
fn test_split_with_sizes_zero_length() {
    let device = Default::default();
    let tensors = TestTensor::<1>::from_data([0, 1, 2], &device);

    let split_tensors = tensors.split_with_sizes(vec![0, 1, 2], 0);
    assert_eq!(split_tensors.len(), 2);

    let expected = [TensorData::from([0]), TensorData::from([1, 2])];

    for (index, tensor) in split_tensors.iter().enumerate() {
        tensor.to_data().assert_eq(&expected[index], false);
    }
}
