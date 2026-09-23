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
    let tensor = TestTensor::<4>::from_data(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );
    let grid = TestTensor::<4>::from_data(
        [[[[0.0, 0.0], [-1.0, 0.25]], [[1.0, 1.0], [0.2, -0.8]]]],
        &device,
    );

    let output = tensor.grid_sample_2d(grid, InterpolateMode::Bilinear);

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
    let tensor = TestTensor::<4>::from_data(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );
    let grid = TestTensor::<4>::from_data(
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
    let tensor = TestTensor::<4>::from_data(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );
    let grid = TestTensor::<4>::from_data([[[[0.0, -2.0]]]], &device);

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
    let tensor = TestTensor::<4>::from_data(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );
    let grid = TestTensor::<4>::from_data([[[[0.0, -2.0]]]], &device);

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

/// Tests bilinear interpolation with reflection padding.
#[test]
fn should_pad_reflection_grid_sample_2d() {
    let device = Default::default();
    let tensor = TestTensor::<4>::from_data(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );
    let grid = TestTensor::<4>::from_data(
        [[[[0.0, 0.0], [-1.0, 0.25]], [[1.0, 1.0], [0.2, -0.8]]]],
        &device,
    );

    let options = GridSampleOptions::new(InterpolateMode::Bilinear)
        .with_padding_mode(GridSamplePaddingMode::Reflection);
    let output = tensor.grid_sample_2d(grid, options);

    // Expected values computed with PyTorch F.grid_sample(mode='bilinear', padding_mode='reflection', align_corners=False)
    let expected = TensorData::from([[[[4.0, 4.125], [8.0, 1.3]]]]);
    output
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

/// Tests nearest grid_sample_2d on coordinates that fall exactly halfway between two pixels.
///
/// PyTorch picks the pixel with `nearbyint`, so ties go to the even index. For a 2x4 input with
/// align_corners=false:
/// - (-0.5, -0.5) maps to pixel (0.5, 0.0) -> x rounds to 0 -> 0.0
/// - (0.0, -0.5) maps to pixel (1.5, 0.0) -> x rounds to 2 -> 2.0
/// - (0.5, -0.5) maps to pixel (2.5, 0.0) -> x rounds to 2 -> 2.0
/// - (-0.75, 0.0) maps to pixel (0.0, 0.5) -> y rounds to 0 -> 0.0
///
/// Other backends do not implement nearest grid sampling yet.
#[cfg(feature = "flex")]
#[test]
fn should_grid_sample_2d_nearest_round_half_to_even() {
    let device = Default::default();
    let tensor =
        TestTensor::<4>::from_data([[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]], &device);
    let grid = TestTensor::<4>::from_data(
        [[[[-0.5, -0.5], [0.0, -0.5], [0.5, -0.5], [-0.75, 0.0]]]],
        &device,
    );

    let output = tensor.grid_sample_2d(grid, InterpolateMode::Nearest);

    // Expected values follow PyTorch's nearbyint rule for grid_sample(mode='nearest')
    let expected = TensorData::from([[[[0.0, 2.0, 2.0, 0.0]]]]);
    output
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

/// A grid held component-first, `[N, 2, H, W]`, and moved to the trailing position with a
/// `permute` view has the required shape but a non-unit component stride. It must sample the
/// same points as the contiguous grid.
#[test]
fn should_grid_sample_2d_permuted_grid_view() {
    let device = Default::default();
    let tensor = TestTensor::<4>::from_data(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
        &device,
    );
    let grid = TestTensor::<4>::from_data(
        [[[[0.0, 0.0], [-1.0, 0.25]], [[1.0, 1.0], [0.2, -0.8]]]],
        &device,
    );
    // The same four points, stored as all x then all y.
    let permuted = TestTensor::<4>::from_data(
        [[[[0.0, -1.0], [1.0, 0.2]], [[0.0, 0.25], [1.0, -0.8]]]],
        &device,
    )
    .permute([0, 2, 3, 1]);
    assert_eq!(permuted.dims(), grid.dims());

    let options = GridSampleOptions::new(InterpolateMode::Bilinear)
        .with_padding_mode(GridSamplePaddingMode::Reflection);
    let expected = tensor
        .clone()
        .grid_sample_2d(grid, options.clone())
        .to_data();
    let output = tensor.grid_sample_2d(permuted, options);
    output
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

/// Both tensors must be rank 4.
#[test]
#[should_panic(expected = "=== Tensor Operation Error ===")]
fn should_grid_sample_2d_reject_wrong_rank() {
    let device = Default::default();
    let tensor = TestTensor::<3>::zeros([1, 2, 2], &device);
    let grid = TestTensor::<3>::zeros([2, 2, 2], &device);

    let _ = tensor.grid_sample_2d(grid, GridSampleOptions::default());
}

/// The grid's last dimension must hold exactly the two `(x, y)` coordinates.
#[test]
#[should_panic(expected = "=== Tensor Operation Error ===")]
fn should_grid_sample_2d_reject_wrong_coordinate_count() {
    let device = Default::default();
    let tensor = TestTensor::<4>::zeros([1, 1, 2, 2], &device);
    let grid = TestTensor::<4>::zeros([1, 2, 2, 3], &device);

    let _ = tensor.grid_sample_2d(grid, GridSampleOptions::default());
}

/// The output is allocated over the input's batch, so a grid with a different batch extent is
/// rejected up front rather than read out of bounds or silently truncated.
#[test]
#[should_panic(expected = "=== Tensor Operation Error ===")]
fn should_grid_sample_2d_reject_batch_mismatch() {
    let device = Default::default();
    let tensor = TestTensor::<4>::zeros([2, 1, 2, 2], &device);
    let grid = TestTensor::<4>::zeros([1, 2, 2, 2], &device);

    let _ = tensor.grid_sample_2d(grid, GridSampleOptions::default());
}
