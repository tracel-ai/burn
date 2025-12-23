use super::*;
use burn_tensor::{
    TensorData, Tolerance,
    ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode},
};

/// Tests grid_sample_2d with default options (align_corners=false, zeros padding).
///
/// For a 3x3 input with grid coordinates:
/// - (0.0, 0.0) maps to pixel (1.0, 1.0) -> center pixel = 4.0
/// - (-1.0, 0.25) maps to pixel (-0.5, 1.375) -> partially out of bounds
/// - (1.0, 1.0) maps to pixel (2.5, 2.5) -> corner, partially out of bounds
/// - (0.2, -0.8) maps to pixel (1.3, 0.3) -> interpolates around center-top
#[test]
fn should_grid_sample_2d_default() {
    let device = Default::default();
    let tensor = TestTensor::<4>::from_floats(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );
    let grid = TestTensor::<4>::from_floats(
        [[[[0.0, 0.0], [-1.0, 0.25]], [[1.0, 1.0], [0.2, -0.8]]]],
        &device,
    );

    let output = tensor.grid_sample_2d(grid, GridSampleOptions::default());

    // Expected values computed with PyTorch grid_sample(align_corners=False, padding_mode='zeros')
    let expected = TensorData::from([[[[4.0, 2.0625], [2.0, 1.04]]]]);
    output
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

/// Tests grid_sample_2d with align_corners=true and border padding.
///
/// This is the original Burn semantics before the API change.
#[test]
fn should_grid_sample_2d_align_corners_border() {
    let device = Default::default();
    let tensor = TestTensor::<4>::from_floats(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );
    let grid = TestTensor::<4>::from_floats(
        [[[[0.0, 0.0], [-1.0, 0.25]], [[1.0, 1.0], [0.2, -0.8]]]],
        &device,
    );

    let options = GridSampleOptions::new(InterpolateMode::Bilinear)
        .with_padding_mode(GridSamplePaddingMode::Border)
        .with_align_corners(true);
    let output = tensor.grid_sample_2d(grid, options);

    // Expected values computed with PyTorch grid_sample(align_corners=True, padding_mode='border')
    let expected = TensorData::from([[[[4.0, 3.75], [8.0, 1.8]]]]);
    output
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

/// Tests out-of-bounds grid coordinates with zeros padding.
/// Grid coordinate (0.0, -2.0) maps to pixel (1.0, -2.5) which is completely out of bounds.
#[test]
fn should_pad_zeros_grid_sample_2d() {
    let device = Default::default();
    let tensor = TestTensor::<4>::from_floats(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );
    let grid = TestTensor::<4>::from_floats([[[[0.0, -2.0]]]], &device);

    let output = tensor.grid_sample_2d(grid, GridSampleOptions::default());

    // With zeros padding, out-of-bounds samples return 0
    let expected = TensorData::from([[[[0.0]]]]);
    output
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

/// Tests out-of-bounds grid coordinates with border padding.
#[test]
fn should_pad_border_grid_sample_2d() {
    let device = Default::default();
    let tensor = TestTensor::<4>::from_floats(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );
    let grid = TestTensor::<4>::from_floats([[[[0.0, -2.0]]]], &device);

    let options = GridSampleOptions::new(InterpolateMode::Bilinear)
        .with_padding_mode(GridSamplePaddingMode::Border);
    let output = tensor.grid_sample_2d(grid, options);

    // With border padding, out-of-bounds coordinates are clamped to border
    // Grid (0.0, -2.0) with align_corners=false: pixel (1.0, -2.5) -> clamped to (1.0, 0.0) = 1.0
    let expected = TensorData::from([[[[1.0]]]]);
    output
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
