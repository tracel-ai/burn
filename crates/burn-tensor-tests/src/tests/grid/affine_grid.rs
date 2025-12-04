use crate::*;
use burn_tensor::grid::affine_grid_2d;

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
