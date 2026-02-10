use super::*;
use burn_tensor::TensorData;

#[test]
fn test_gather_nd_2d_k_equals_d() {
    // Gather single elements from a 2D tensor (K=D=2)
    let device = Default::default();
    // data shape: [3, 3]
    let data = TestTensor::<2>::from_floats(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        &device,
    );
    // indices shape: [3, 2] => K=2, M=2, DV = M-1+D-K = 1+2-2 = 1
    // Each row is [row, col]
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [1, 2], [2, 0]], &device);

    let output: TestTensor<1> = data.gather_nd(indices);

    output
        .into_data()
        .assert_eq(&TensorData::from([1.0, 5.0, 6.0]), false);
}

#[test]
fn test_gather_nd_3d_k1_slices() {
    // Gather 2D slices from a 3D tensor (K=1)
    let device = Default::default();
    // data shape: [2, 3, 4]
    let data = TestTensor::<3>::from_floats(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            [
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
        ],
        &device,
    );
    // indices shape: [2, 1] => K=1, M=2, DV = 1 + 3 - 1 = 3
    // Each index selects a whole 2D slice along dim 0
    let indices = TestTensorInt::<2>::from_ints([[1], [0]], &device);

    let output: TestTensor<3> = data.gather_nd(indices);

    output.into_data().assert_eq(
        &TensorData::from([
            [
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
        ]),
        false,
    );
}

#[test]
fn test_gather_nd_3d_k_equals_d() {
    // Gather scalar elements from a 3D tensor (K=D=3)
    let device = Default::default();
    let data = TestTensor::<3>::from_floats(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ],
        &device,
    );
    // indices shape: [2, 3] => K=3, M=2, DV = 1+3-3 = 1
    let indices = TestTensorInt::<2>::from_ints([[0, 0, 1], [1, 1, 0]], &device);

    let output: TestTensor<1> = data.gather_nd(indices);

    output
        .into_data()
        .assert_eq(&TensorData::from([1.0, 6.0]), false);
}

#[test]
fn test_gather_nd_batch() {
    // Batch dimension in indices
    let device = Default::default();
    // data shape: [2, 3]
    let data = TestTensor::<2>::from_floats([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], &device);
    // indices shape: [3, 2] => K=2, M=2, DV = 1+2-2 = 1
    let indices = TestTensorInt::<2>::from_ints([[0, 0], [0, 2], [1, 1]], &device);

    let output: TestTensor<1> = data.gather_nd(indices);

    output
        .into_data()
        .assert_eq(&TensorData::from([10.0, 30.0, 50.0]), false);
}

#[test]
fn test_scatter_nd_assign_2d() {
    let device = Default::default();
    // data shape: [3, 3], initialized to zeros
    let data = TestTensor::<2>::zeros([3, 3], &device);
    // indices shape: [2, 2] => K=2, M=2, DV = 1+2-2 = 1
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [2, 0]], &device);
    // values shape: [2] (scalar per index tuple)
    let values = TestTensor::<1>::from_floats([10.0, 20.0], &device);

    let output = data.scatter_nd(indices, values);

    output.into_data().assert_eq(
        &TensorData::from([[0.0, 10.0, 0.0], [0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        false,
    );
}

#[test]
fn test_scatter_nd_add_2d() {
    let device = Default::default();
    let data = TestTensor::<2>::from_floats(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        &device,
    );
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [2, 0]], &device);
    let values = TestTensor::<1>::from_floats([10.0, 20.0], &device);

    let output = data.scatter_nd_add(indices, values);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 11.0, 1.0], [1.0, 1.0, 1.0], [21.0, 1.0, 1.0]]),
        false,
    );
}

#[test]
fn test_scatter_nd_add_duplicate_indices() {
    let device = Default::default();
    let data = TestTensor::<2>::zeros([2, 3], &device);
    // Both index tuples point to the same location
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [0, 1]], &device);
    let values = TestTensor::<1>::from_floats([5.0, 3.0], &device);

    let output = data.scatter_nd_add(indices, values);

    // Values should accumulate: 5.0 + 3.0 = 8.0
    output.into_data().assert_eq(
        &TensorData::from([[0.0, 8.0, 0.0], [0.0, 0.0, 0.0]]),
        false,
    );
}

#[test]
fn test_scatter_nd_3d_slices() {
    // Scatter 1D slices into a 3D tensor (K=1)
    let device = Default::default();
    // data shape: [2, 3]
    let data = TestTensor::<2>::zeros([2, 3], &device);
    // indices shape: [2, 1] => K=1, M=2, DV = 1+2-1 = 2
    // Each index selects a row
    let indices = TestTensorInt::<2>::from_ints([[0], [1]], &device);
    // values shape: [2, 3]
    let values =
        TestTensor::<2>::from_floats([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], &device);

    let output = data.scatter_nd(indices, values);

    output.into_data().assert_eq(
        &TensorData::from([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        false,
    );
}

#[test]
fn test_scatter_nd_mul() {
    let device = Default::default();
    let data = TestTensor::<2>::from_floats(
        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        &device,
    );
    let indices = TestTensorInt::<2>::from_ints([[0, 1], [1, 2]], &device);
    let values = TestTensor::<1>::from_floats([10.0, 2.0], &device);

    let output = data.scatter_nd_mul(indices, values);

    output.into_data().assert_eq(
        &TensorData::from([[2.0, 30.0, 4.0], [5.0, 6.0, 14.0]]),
        false,
    );
}

#[test]
fn test_scatter_nd_min() {
    let device = Default::default();
    let data = TestTensor::<2>::from_floats(
        [[5.0, 10.0], [15.0, 20.0]],
        &device,
    );
    let indices = TestTensorInt::<2>::from_ints([[0, 0], [1, 1]], &device);
    let values = TestTensor::<1>::from_floats([3.0, 25.0], &device);

    let output = data.scatter_nd_min(indices, values);

    // min(5.0, 3.0) = 3.0; min(20.0, 25.0) = 20.0
    output.into_data().assert_eq(
        &TensorData::from([[3.0, 10.0], [15.0, 20.0]]),
        false,
    );
}

#[test]
fn test_scatter_nd_max() {
    let device = Default::default();
    let data = TestTensor::<2>::from_floats(
        [[5.0, 10.0], [15.0, 20.0]],
        &device,
    );
    let indices = TestTensorInt::<2>::from_ints([[0, 0], [1, 1]], &device);
    let values = TestTensor::<1>::from_floats([3.0, 25.0], &device);

    let output = data.scatter_nd_max(indices, values);

    // max(5.0, 3.0) = 5.0; max(20.0, 25.0) = 25.0
    output.into_data().assert_eq(
        &TensorData::from([[5.0, 10.0], [15.0, 25.0]]),
        false,
    );
}

#[test]
fn test_scatter_nd_batch() {
    // Scatter with batch dimensions
    let device = Default::default();
    // data shape: [3, 3]
    let data = TestTensor::<2>::zeros([3, 3], &device);
    // indices shape: [3, 1] => K=1, M=2, DV = 1+2-1 = 2
    let indices = TestTensorInt::<2>::from_ints([[0], [2], [1]], &device);
    // values shape: [3, 3]
    let values = TestTensor::<2>::from_floats(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        &device,
    );

    let output = data.scatter_nd(indices, values);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0], [4.0, 5.0, 6.0]]),
        false,
    );
}
