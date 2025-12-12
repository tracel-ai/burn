use super::*;
use burn_tensor::linalg::trace;

#[test]
fn test_trace_2d_square() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);
    let result = trace::<_, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([15.0], &device); // 1 + 5 + 9 = 15

    assert_eq!(result.to_data(), expected.to_data());
}

#[test]
fn test_trace_2d_rectangular_wide() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device);
    let result = trace::<_, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([7.0], &device); // 1 + 6 = 7

    assert_eq!(result.to_data(), expected.to_data());
}

#[test]
fn test_trace_2d_rectangular_tall() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
    let result = trace::<_, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([5.0], &device); // 1 + 4 = 5

    assert_eq!(result.to_data(), expected.to_data());
}

#[test]
fn test_trace_3d_batch() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        &device,
    );

    let result = trace::<_, 3, 2>(tensor);
    let expected = TestTensor::<2>::from_data([[5.0], [13.0]], &device); // [1+4=5, 5+8=13]

    assert_eq!(result.to_data(), expected.to_data());
}

#[test]
fn test_trace_4d_batch() {
    let device = Default::default();
    let tensor = TestTensor::<4>::from_data(
        [[
            // Batch 0, Channel 0
            [[1.0, 2.0], [3.0, 4.0]],
            // Batch 0, Channel 1
            [[5.0, 6.0], [7.0, 8.0]],
        ]],
        &device,
    );

    let result = trace::<_, 4, 3>(tensor);
    let expected = TestTensor::<3>::from_data([[[5.0], [13.0]]], &device);

    assert_eq!(result.to_data(), expected.to_data());
}

#[test]
fn test_trace_single_element() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[42.0]], &device);
    let result = trace::<_, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([42.0], &device);

    assert_eq!(result.to_data(), expected.to_data());
}

#[test]
fn test_trace_zeros() {
    let device = Default::default();
    let tensor = TestTensor::<2>::zeros([3, 3], &device);
    let result = trace::<_, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([0.0], &device);

    assert_eq!(result.to_data(), expected.to_data());
}

#[test]
fn test_trace_negative_values() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[-1.0, 2.0], [3.0, -4.0]], &device);
    let result = trace::<_, 2, 1>(tensor);
    let expected = TestTensor::<1>::from_data([-5.0], &device); // -1 + (-4) = -5

    assert_eq!(result.to_data(), expected.to_data());
}

#[test]
#[should_panic]
fn test_trace_1d_should_panic() {
    let device = Default::default();
    // 1D tensor should panic - trace requires at least 2 dimensions
    let tensor = TestTensor::<1>::from_data([1.0, 2.0, 3.0], &device);
    let _result = trace::<_, 1, 0>(tensor);
}

#[test]
#[should_panic]
fn test_trace_wrong_output_rank_should_panic() {
    let device = Default::default();
    // Providing wrong output rank should panic
    let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let _result = trace::<_, 2, 2>(tensor); // Should be 2,1 not 2,2
}
