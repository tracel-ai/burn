use super::*;
use burn_tensor::{TensorData, ops::PadMode};

#[test]
fn padding_constant_2d_test() {
    let unpadded_floats: [[f32; 3]; 2] = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
    let tensor = TestTensor::<2>::from(unpadded_floats);

    let padded_tensor = tensor.pad((2, 2, 2, 2), 1.1);

    let expected = TensorData::from([
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 0.0, 1.0, 2.0, 1.1, 1.1],
        [1.1, 1.1, 3.0, 4.0, 5.0, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    ]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_constant_4d_test() {
    let unpadded_floats = [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]];
    let tensor = TestTensor::<4>::from(unpadded_floats);

    let padded_tensor = tensor.pad((2, 2, 2, 2), 1.1);

    let expected = TensorData::from([[[
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 0.0, 1.0, 1.1, 1.1],
        [1.1, 1.1, 2.0, 3.0, 1.1, 1.1],
        [1.1, 1.1, 4.0, 5.0, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    ]]]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_constant_asymmetric_test() {
    let unpadded_floats = [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]];
    let tensor = TestTensor::<4>::from(unpadded_floats);

    let padded_tensor = tensor.pad((2, 1, 4, 3), 1.1);

    let expected = TensorData::from([[[
        [1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 0.0, 1.0, 1.1],
        [1.1, 1.1, 2.0, 3.0, 1.1],
        [1.1, 1.1, 4.0, 5.0, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1],
    ]]]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_reflect_2d_test() {
    // Test reflect padding on a 2D tensor
    // Input: [[1, 2, 3], [4, 5, 6]]
    // With padding (1, 1, 1, 1):
    // - Top: reflect row 1 -> [4, 5, 6]
    // - Bottom: reflect row 0 -> [1, 2, 3]
    // - Left: reflect col 1
    // - Right: reflect col 1
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let padded_tensor = tensor.pad((1, 1, 1, 1), PadMode::Reflect);

    // Expected: reflect excludes the edge value
    // Before padding height: [[1,2,3], [4,5,6]]
    // After top pad (reflect row at index 1): [[4,5,6], [1,2,3], [4,5,6]]
    // After bottom pad (reflect row at index 1 from end): [[4,5,6], [1,2,3], [4,5,6], [1,2,3]]
    // Then pad width similarly
    let expected = TensorData::from([
        [5.0, 4.0, 5.0, 6.0, 5.0],
        [2.0, 1.0, 2.0, 3.0, 2.0],
        [5.0, 4.0, 5.0, 6.0, 5.0],
        [2.0, 1.0, 2.0, 3.0, 2.0],
    ]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_reflect_width_only_test() {
    // Test reflect padding on width dimension only
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0]]);

    let padded_tensor = tensor.pad((2, 2, 0, 0), PadMode::Reflect);

    // Input: [1, 2, 3, 4]
    // Reflect left 2: take indices [1, 2] = [2, 3], flip = [3, 2]
    // Reflect right 2: take indices [1, 2] from end = [2, 3], flip = [3, 2]
    // Result: [3, 2, 1, 2, 3, 4, 3, 2]
    let expected = TensorData::from([[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_reflect_4d_test() {
    // Test reflect padding on 4D tensor (common for images: NCHW)
    let tensor = TestTensor::<4>::from([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]);

    let padded_tensor = tensor.pad((1, 1, 1, 1), PadMode::Reflect);

    let expected = TensorData::from([[[
        [5.0, 4.0, 5.0, 6.0, 5.0],
        [2.0, 1.0, 2.0, 3.0, 2.0],
        [5.0, 4.0, 5.0, 6.0, 5.0],
        [8.0, 7.0, 8.0, 9.0, 8.0],
        [5.0, 4.0, 5.0, 6.0, 5.0],
    ]]]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_edge_2d_test() {
    // Test edge padding on a 2D tensor
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let padded_tensor = tensor.pad((1, 1, 1, 1), PadMode::Edge);

    // Edge padding replicates the boundary values
    let expected = TensorData::from([
        [1.0, 1.0, 2.0, 3.0, 3.0],
        [1.0, 1.0, 2.0, 3.0, 3.0],
        [4.0, 4.0, 5.0, 6.0, 6.0],
        [4.0, 4.0, 5.0, 6.0, 6.0],
    ]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_edge_width_only_test() {
    // Test edge padding on width dimension only
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0]]);

    let padded_tensor = tensor.pad((2, 3, 0, 0), PadMode::Edge);

    // Input: [1, 2, 3, 4]
    // Left 2: [1, 1]
    // Right 3: [4, 4, 4]
    // Result: [1, 1, 1, 2, 3, 4, 4, 4, 4]
    let expected = TensorData::from([[1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0]]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_edge_4d_test() {
    // Test edge padding on 4D tensor
    let tensor = TestTensor::<4>::from([[[[1.0, 2.0], [3.0, 4.0]]]]);

    let padded_tensor = tensor.pad((1, 1, 1, 1), PadMode::Edge);

    let expected = TensorData::from([[[
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [3.0, 3.0, 4.0, 4.0],
        [3.0, 3.0, 4.0, 4.0],
    ]]]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_constant_default_test() {
    // Test default PadMode (Constant with 0.0)
    let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

    let padded_tensor = tensor.pad((1, 1, 1, 1), PadMode::default());

    let expected = TensorData::from([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 3.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_reflect_max_valid_test() {
    // Test reflect padding at maximum valid size (dim_size - 1)
    // For a 4-element dimension, max valid padding is 3
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0, 4.0]]);

    // Padding of 3 on left is valid for width=4 (3 < 4)
    let padded_tensor = tensor.pad((3, 3, 0, 0), PadMode::Reflect);

    // Input: [1, 2, 3, 4]
    // Reflect left 3: take indices [1, 2, 3] = [2, 3, 4], flip = [4, 3, 2]
    // Reflect right 3: take indices [0, 1, 2] = [1, 2, 3], flip = [3, 2, 1]
    // Result: [4, 3, 2, 1, 2, 3, 4, 3, 2, 1]
    let expected = TensorData::from([[4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_reflect_asymmetric_test() {
    // Test asymmetric reflect padding
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    // Asymmetric padding: left=2, right=1, top=1, bottom=2
    let padded_tensor = tensor.pad((2, 1, 1, 2), PadMode::Reflect);

    let expected = TensorData::from([
        [6.0, 5.0, 4.0, 5.0, 6.0, 5.0],
        [3.0, 2.0, 1.0, 2.0, 3.0, 2.0],
        [6.0, 5.0, 4.0, 5.0, 6.0, 5.0],
        [9.0, 8.0, 7.0, 8.0, 9.0, 8.0],
        [6.0, 5.0, 4.0, 5.0, 6.0, 5.0],
        [3.0, 2.0, 1.0, 2.0, 3.0, 2.0],
    ]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic(expected = "Reflect padding")]
fn padding_reflect_exceeds_dimension_test() {
    // Test that reflect padding panics when padding >= dim_size
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0]]);

    // Padding of 3 on width=3 should panic (3 >= 3, need padding < dim_size)
    let _ = tensor.pad((3, 0, 0, 0), PadMode::Reflect);
}

#[test]
fn padding_edge_asymmetric_test() {
    // Test asymmetric edge padding
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    // Asymmetric padding: left=2, right=1, top=3, bottom=1
    let padded_tensor = tensor.pad((2, 1, 3, 1), PadMode::Edge);

    let expected = TensorData::from([
        [1.0, 1.0, 1.0, 2.0, 3.0, 3.0],
        [1.0, 1.0, 1.0, 2.0, 3.0, 3.0],
        [1.0, 1.0, 1.0, 2.0, 3.0, 3.0],
        [1.0, 1.0, 1.0, 2.0, 3.0, 3.0],
        [4.0, 4.0, 4.0, 5.0, 6.0, 6.0],
        [4.0, 4.0, 4.0, 5.0, 6.0, 6.0],
    ]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_zero_padding_test() {
    // Test that zero padding returns the original tensor unchanged
    let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

    let padded_constant = tensor.clone().pad((0, 0, 0, 0), PadMode::Constant(5.0));
    let padded_reflect = tensor.clone().pad((0, 0, 0, 0), PadMode::Reflect);
    let padded_edge = tensor.clone().pad((0, 0, 0, 0), PadMode::Edge);

    let expected = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    padded_constant.into_data().assert_eq(&expected, false);
    padded_reflect.into_data().assert_eq(&expected, false);
    padded_edge.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_empty_tensor_constant_test() {
    // Test constant padding on an empty tensor (zero-sized dimension)
    // This should work - creates a tensor filled with the constant value
    let tensor: TestTensor<2> = TestTensor::empty([0, 3], &Default::default());

    // Padding an empty height dimension with constant should create a tensor of just padding
    let padded = tensor.pad((0, 0, 2, 2), 1.0);

    // Result should be 4x3 (0 + 2 + 2 = 4 rows)
    assert_eq!(padded.dims(), [4, 3]);

    let expected = TensorData::from([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]);
    padded.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic(expected = "edge padding")]
fn padding_empty_tensor_edge_panics_test() {
    // Test that edge padding panics on empty tensor
    let tensor: TestTensor<2> = TestTensor::empty([0, 3], &Default::default());

    // Edge padding on zero-sized dimension should panic
    let _ = tensor.pad((0, 0, 1, 1), PadMode::Edge);
}

#[test]
#[should_panic(expected = "Reflect padding")]
fn padding_empty_tensor_reflect_panics_test() {
    // Test that reflect padding panics on empty tensor
    let tensor: TestTensor<2> = TestTensor::empty([0, 3], &Default::default());

    // Reflect padding on zero-sized dimension should panic
    let _ = tensor.pad((0, 0, 1, 1), PadMode::Reflect);
}

// --- Tests for N-dimensional padding using (before, after) pairs ---

#[test]
fn padding_constant_pairs_2d_test() {
    // Same as padding_constant_2d_test but using the new pairs API
    let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    // [(row_before, row_after), (col_before, col_after)]
    let padded_tensor = tensor.pad([(2, 2), (2, 2)], 1.1);

    let expected = TensorData::from([
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 0.0, 1.0, 2.0, 1.1, 1.1],
        [1.1, 1.1, 3.0, 4.0, 5.0, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    ]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_constant_single_dim_test() {
    // Pad only the last dimension
    let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

    let padded_tensor = tensor.pad([(1, 1)], 0.0);

    let expected = TensorData::from([[0.0, 1.0, 2.0, 0.0], [0.0, 3.0, 4.0, 0.0]]);
    padded_tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_constant_all_dims_4d_test() {
    // Pad all 4 dimensions of a 4D tensor (batch, channel, height, width)
    // Input: shape [1, 1, 2, 2]
    let tensor = TestTensor::<4>::from([[[[1.0, 2.0], [3.0, 4.0]]]]);

    // Pad: batch(1,1), channel(1,1), height(0,0), width(0,0)
    let padded = tensor.pad([(1, 1), (1, 1), (0, 0), (0, 0)], 0.0);

    // Shape should be [3, 3, 2, 2]
    assert_eq!(padded.dims(), [3, 3, 2, 2]);

    let expected = TensorData::from([
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
    ]);
    padded.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_constant_batch_dim_only_test() {
    // Pad only the batch dimension of a 3D tensor [N, H, W]
    let tensor = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]]]);

    // 3 pairs for 3 dims: batch(1,1), height(0,0), width(0,0)
    let padded = tensor.pad([(1, 1), (0, 0), (0, 0)], -1.0);

    assert_eq!(padded.dims(), [3, 2, 2]);

    let expected = TensorData::from([
        [[-1.0, -1.0], [-1.0, -1.0]],
        [[1.0, 2.0], [3.0, 4.0]],
        [[-1.0, -1.0], [-1.0, -1.0]],
    ]);
    padded.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_reflect_pairs_test() {
    // Reflect padding using pairs API
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    let padded = tensor.pad([(1, 1), (1, 1)], PadMode::Reflect);

    let expected = TensorData::from([
        [5.0, 4.0, 5.0, 6.0, 5.0],
        [2.0, 1.0, 2.0, 3.0, 2.0],
        [5.0, 4.0, 5.0, 6.0, 5.0],
        [8.0, 7.0, 8.0, 9.0, 8.0],
        [5.0, 4.0, 5.0, 6.0, 5.0],
    ]);
    padded.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_edge_pairs_test() {
    // Edge padding using pairs API
    let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);

    let padded = tensor.pad([(1, 1), (1, 1)], PadMode::Edge);

    let expected = TensorData::from([
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [3.0, 3.0, 4.0, 4.0],
        [3.0, 3.0, 4.0, 4.0],
    ]);
    padded.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_reflect_batch_dim_3d_test() {
    // Reflect pad the batch dimension of a 3D tensor [N, H, W]
    // Input shape: [3, 1, 2] - 3 batches, 1 row, 2 cols
    let tensor = TestTensor::<3>::from([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]);

    // Pad batch dim with reflect(1, 1), no spatial padding
    let padded = tensor.pad([(1, 1), (0, 0), (0, 0)], PadMode::Reflect);

    assert_eq!(padded.dims(), [5, 1, 2]);

    // Reflect on batch: [3,4] [1,2] [3,4] [5,6] [3,4]
    let expected = TensorData::from([
        [[3.0, 4.0]],
        [[1.0, 2.0]],
        [[3.0, 4.0]],
        [[5.0, 6.0]],
        [[3.0, 4.0]],
    ]);
    padded.into_data().assert_eq(&expected, false);
}

#[test]
fn padding_edge_batch_dim_3d_test() {
    // Edge pad the batch dimension of a 3D tensor
    let tensor = TestTensor::<3>::from([[[1.0, 2.0]], [[3.0, 4.0]]]);

    let padded = tensor.pad([(2, 1), (0, 0), (0, 0)], PadMode::Edge);

    assert_eq!(padded.dims(), [5, 1, 2]);

    let expected = TensorData::from([
        [[1.0, 2.0]],
        [[1.0, 2.0]],
        [[1.0, 2.0]],
        [[3.0, 4.0]],
        [[3.0, 4.0]],
    ]);
    padded.into_data().assert_eq(&expected, false);
}

#[test]
#[should_panic(expected = "Padding has")]
fn padding_too_many_pairs_panics_test() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0]]);

    // 3 pairs for a 2D tensor should panic
    let _ = tensor.pad([(1, 1), (1, 1), (1, 1)], 0.0);
}
