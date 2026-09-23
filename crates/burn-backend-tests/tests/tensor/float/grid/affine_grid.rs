use super::*;
use burn_tensor::grid::{affine_grid_2d, affine_grid_3d};

fn create_identity_transform(batch_size: usize) -> TestTensor<3> {
    // Identity affine transform (batch_size, 2, 3)
    TestTensor::<3>::from([[[1f32, 0., 0.], [0., 1., 0.]]]).expand([batch_size, 2, 3])
}

#[test]
fn test_affine_grid_identity() {
    let batch_size = 1;
    let channels = 1;
    let height = 2;
    let width = 2;

    let transform = create_identity_transform(batch_size);

    let output = affine_grid_2d(transform, [batch_size, channels, height, width]);

    // Expected normalized coords:
    // [-1, -1], [ 1,-1]
    // [-1,  1], [ 1, 1]
    let expected = TestTensor::<4>::from([[
        [[-1f32, -1f32], [1f32, -1f32]],
        [[-1f32, 1f32], [1f32, 1f32]],
    ]]);

    output.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn test_affine_grid_scaling() {
    let batch_size = 1;
    let channels = 1;
    let height = 2;
    let width = 2;

    let scale = 2.0f32;
    let transform = TestTensor::<3>::from([[[scale, 0., 0.], [0., scale, 0.]]]);

    let output = affine_grid_2d(transform, [batch_size, channels, height, width]);

    // Expect scaled coordinates from normalized grid, so coords * 2
    let expected = TestTensor::<4>::from([[
        [[-2f32, -2f32], [2f32, -2f32]],
        [[-2f32, 2f32], [2f32, 2f32]],
    ]]);

    output.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn test_affine_grid_translation() {
    let batch_size = 1;
    let channels = 1;
    let height = 2;
    let width = 2;

    // Translate by 0.5 in x and -0.5 in y (normalized coords)
    let tx = 0.5f32;
    let ty = -0.5f32;

    let transform = TestTensor::<3>::from([[[1.0, 0.0, tx], [0.0, 1.0, ty]]]);

    let output = affine_grid_2d(transform, [batch_size, channels, height, width]);

    // Expected coordinates:
    // Original normalized coords are [-1,1] in x and y
    // After translation, each coordinate shifts by tx and ty
    // So points become:
    // [-1 + 0.5, -1 - 0.5] = [-0.5, -1.5]
    // [ 1 + 0.5, -1 - 0.5] = [1.5, -1.5]
    // [-1 + 0.5,  1 - 0.5] = [-0.5, 0.5]
    // [ 1 + 0.5,  1 - 0.5] = [1.5, 0.5]

    let expected = TestTensor::<4>::from([[
        [[-0.5f32, -1.5f32], [1.5f32, -1.5f32]],
        [[-0.5f32, 0.5f32], [1.5f32, 0.5f32]],
    ]]);

    output.into_data().assert_eq(&expected.into_data(), false);
}

fn create_identity_transform_3d(batch_size: usize) -> TestTensor<3> {
    // Identity affine transform (batch_size, 3, 4)
    TestTensor::<3>::from([[[1f32, 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]])
        .expand([batch_size, 3, 4])
}

#[test]
fn test_affine_grid_3d_identity() {
    let batch_size = 1;
    let channels = 1;
    let depth = 2;
    let height = 2;
    let width = 2;

    let transform = create_identity_transform_3d(batch_size);

    let output = affine_grid_3d(transform, [batch_size, channels, depth, height, width]);

    // Expected normalized (x, y, z) coords, x fastest:
    // z = -1: [-1, -1, -1], [ 1, -1, -1]
    //         [-1,  1, -1], [ 1,  1, -1]
    // z =  1: [-1, -1,  1], [ 1, -1,  1]
    //         [-1,  1,  1], [ 1,  1,  1]
    let expected = TestTensor::<5>::from([[
        [
            [[-1f32, -1., -1.], [1., -1., -1.]],
            [[-1., 1., -1.], [1., 1., -1.]],
        ],
        [
            [[-1., -1., 1.], [1., -1., 1.]],
            [[-1., 1., 1.], [1., 1., 1.]],
        ],
    ]]);

    output.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn test_affine_grid_3d_scaling() {
    let batch_size = 1;
    let channels = 1;
    let depth = 2;
    let height = 2;
    let width = 2;

    // A different scale on every axis, so a swapped axis is visible
    let transform =
        TestTensor::<3>::from([[[2f32, 0., 0., 0.], [0., 3., 0., 0.], [0., 0., 4., 0.]]]);

    let output = affine_grid_3d(transform, [batch_size, channels, depth, height, width]);

    let expected = TestTensor::<5>::from([[
        [
            [[-2f32, -3., -4.], [2., -3., -4.]],
            [[-2., 3., -4.], [2., 3., -4.]],
        ],
        [
            [[-2., -3., 4.], [2., -3., 4.]],
            [[-2., 3., 4.], [2., 3., 4.]],
        ],
    ]]);

    output.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn test_affine_grid_3d_translation() {
    let batch_size = 1;
    let channels = 1;
    let depth = 2;
    let height = 2;
    let width = 2;

    // Translate by 0.5 in x, -0.5 in y and 0.25 in z (normalized coords)
    let transform =
        TestTensor::<3>::from([[[1f32, 0., 0., 0.5], [0., 1., 0., -0.5], [0., 0., 1., 0.25]]]);

    let output = affine_grid_3d(transform, [batch_size, channels, depth, height, width]);

    let expected = TestTensor::<5>::from([[
        [
            [[-0.5f32, -1.5, -0.75], [1.5, -1.5, -0.75]],
            [[-0.5, 0.5, -0.75], [1.5, 0.5, -0.75]],
        ],
        [
            [[-0.5, -1.5, 1.25], [1.5, -1.5, 1.25]],
            [[-0.5, 0.5, 1.25], [1.5, 0.5, 1.25]],
        ],
    ]]);

    output.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn test_affine_grid_3d_batched() {
    let batch_size = 2;
    let channels = 3;
    let depth = 2;
    let height = 2;
    let width = 2;

    // Batch 0 is the identity; batch 1 shears x by z and translates y
    let transform = TestTensor::<3>::from([
        [[1f32, 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
        [[1., 0., 0.5, 0.], [0., 1., 0., 0.25], [0., 0., 1., 0.]],
    ]);

    let output = affine_grid_3d(transform, [batch_size, channels, depth, height, width]);
    assert_eq!(output.dims(), [batch_size, depth, height, width, 3]);

    // Batch 1: grid_x = x + 0.5 z, grid_y = y + 0.25, grid_z = z
    let expected = TestTensor::<5>::from([
        [
            [
                [[-1f32, -1., -1.], [1., -1., -1.]],
                [[-1., 1., -1.], [1., 1., -1.]],
            ],
            [
                [[-1., -1., 1.], [1., -1., 1.]],
                [[-1., 1., 1.], [1., 1., 1.]],
            ],
        ],
        [
            [
                [[-1.5, -0.75, -1.], [0.5, -0.75, -1.]],
                [[-1.5, 1.25, -1.], [0.5, 1.25, -1.]],
            ],
            [
                [[-0.5, -0.75, 1.], [1.5, -0.75, 1.]],
                [[-0.5, 1.25, 1.], [1.5, 1.25, 1.]],
            ],
        ],
    ]);

    output.into_data().assert_eq(&expected.into_data(), false);
}

#[test]
fn test_affine_grid_3d_singleton_axis() {
    let batch_size = 1;
    let channels = 1;
    let depth = 1;
    let height = 2;
    let width = 3;

    let transform = create_identity_transform_3d(batch_size);

    let output = affine_grid_3d(transform, [batch_size, channels, depth, height, width]);

    // A single slice sits at the center of the z range, while the other axes still span it.
    let expected = TestTensor::<5>::from([[[
        [[-1f32, -1., 0.], [0., -1., 0.], [1., -1., 0.]],
        [[-1., 1., 0.], [0., 1., 0.], [1., 1., 0.]],
    ]]]);

    output.into_data().assert_eq(&expected.into_data(), false);
}
