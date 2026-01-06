use super::*;
use burn_tensor::Tensor;
use burn_tensor::Tolerance;

#[test]
fn test_cross_product() {
    let device = Default::default();
    // Test with well-known orthogonal vectors for clearer validation
    let a = Tensor::<TestBackend, 2>::from_data([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], &device);
    let b = Tensor::<TestBackend, 2>::from_data([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], &device);

    let result = a.cross(b, 1);
    // For orthogonal unit vectors:
    // i × j = k
    // j × k = i
    let expected = Tensor::<TestBackend, 2>::from_data([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], &device);

    // Use Tolerance for floating-point comparisons
    let tolerance = Tolerance::<FloatElem>::default();
    result
        .to_data()
        .assert_approx_eq(&expected.to_data(), tolerance);
}

#[test]
fn test_cross_product_zeros() {
    let device = Default::default();
    // Test cross product with zero vector - should always give zero vector
    let a = Tensor::<TestBackend, 2>::from_data([[2.0, 3.0, 4.0]], &device);
    let b = Tensor::<TestBackend, 2>::zeros([1, 3], &device);

    let result = a.cross(b, 1);
    let expected = Tensor::<TestBackend, 2>::zeros([1, 3], &device);

    // For zeros, we can use exact equality or a very tight tolerance
    let tolerance = Tolerance::<FloatElem>::default();
    result
        .to_data()
        .assert_approx_eq(&expected.to_data(), tolerance);
}

#[test]
fn test_cross_product_batch() {
    let device = Default::default();
    // Test typical cross product computations in batch
    let a = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let b = Tensor::<TestBackend, 2>::from_data([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);

    let result = a.cross(b, 1);
    // Cross products:
    // [1,2,3] × [4,5,6] = [-3,6,-3]
    // [4,5,6] × [7,8,9] = [-3,6,-3]
    let expected =
        Tensor::<TestBackend, 2>::from_data([[-3.0, 6.0, -3.0], [-3.0, 6.0, -3.0]], &device);

    let tolerance = Tolerance::<FloatElem>::default();
    result
        .to_data()
        .assert_approx_eq(&expected.to_data(), tolerance);
}

#[test]
#[should_panic]
fn test_cross_product_invalid_dimension() {
    let device = Default::default();
    let a = Tensor::<TestBackend, 2>::zeros([1, 4], &device);
    let b = Tensor::<TestBackend, 2>::zeros([1, 4], &device);

    let _ = a.cross(b, 1);
}

#[test]
fn test_cross_product_parallel_vectors() {
    let device = Default::default();
    // Test cross product of parallel vectors (should be zero)
    let a = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0]], &device);
    let b = Tensor::<TestBackend, 2>::from_data([[2.0, 4.0, 6.0]], &device); // b = 2 * a

    let result = a.cross(b, 1);
    let expected = Tensor::<TestBackend, 2>::zeros([1, 3], &device);

    let tolerance = Tolerance::<FloatElem>::default();
    result
        .to_data()
        .assert_approx_eq(&expected.to_data(), tolerance);
}

#[test]
fn test_cross_product_3d_tensor() {
    let device = Default::default();
    // Test with 3D tensor (batch of matrices)
    let a = Tensor::<TestBackend, 3>::from_data(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ],
        &device,
    );

    let b = Tensor::<TestBackend, 3>::from_data(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ],
        &device,
    );

    let result = a.cross(b, 2); // Cross on last dimension
    let expected = Tensor::<TestBackend, 3>::from_data(
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[-3.0, 6.0, -3.0], [-3.0, 6.0, -3.0]],
        ],
        &device,
    );

    let tolerance = Tolerance::<FloatElem>::default();
    result
        .to_data()
        .assert_approx_eq(&expected.to_data(), tolerance);
}

// Test to verify that padding doesn't affect results
#[test]
fn test_cross_product_with_padding_awareness() {
    let device = Default::default();
    // Create tensors that would span multiple 4-element blocks
    // This tests that the padding doesn't corrupt adjacent data
    let a = Tensor::<TestBackend, 2>::from_data(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // Two vectors: [1,2,3] and [4,5,6]
        ],
        &device,
    );

    let b = Tensor::<TestBackend, 2>::from_data(
        [
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], // Two vectors: [7,8,9] and [10,11,12]
        ],
        &device,
    );

    // Reshape to have proper 3-element vectors in last dimension
    let a_reshaped = a.reshape([2, 3]);
    let b_reshaped = b.reshape([2, 3]);

    let result = a_reshaped.cross(b_reshaped, 1);

    // Expected cross products:
    // [1,2,3] × [7,8,9] = [-6,12,-6]
    // [4,5,6] × [10,11,12] = [-6,12,-6]
    let expected =
        Tensor::<TestBackend, 2>::from_data([[-6.0, 12.0, -6.0], [-6.0, 12.0, -6.0]], &device);

    let tolerance = Tolerance::<FloatElem>::default();
    result
        .to_data()
        .assert_approx_eq(&expected.to_data(), tolerance);
}
