use super::*;
use burn_tensor::ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode};
use burn_tensor::{Distribution, Tolerance};

fn cases() -> Vec<GridSampleOptions> {
    let mut cases = Vec::new();
    for mode in [InterpolateMode::Bilinear, InterpolateMode::Nearest] {
        for padding_mode in [
            GridSamplePaddingMode::Zeros,
            GridSamplePaddingMode::Border,
            GridSamplePaddingMode::Reflection,
        ] {
            for align_corners in [false, true] {
                cases.push(
                    GridSampleOptions::new(mode.clone())
                        .with_padding_mode(padding_mode)
                        .with_align_corners(align_corners),
                );
            }
        }
    }
    cases
}

fn assert_matches_reference(tensor: TestTensor<5>, grid: TestTensor<5>) {
    let ref_device = ReferenceDevice::new();
    let tensor_ref = TestTensor::<5>::from_data(tensor.to_data(), &ref_device);
    let grid_ref = TestTensor::<5>::from_data(grid.to_data(), &ref_device);

    for options in cases() {
        let output = tensor.clone().grid_sample_3d(grid.clone(), options.clone());
        let output_ref = tensor_ref
            .clone()
            .grid_sample_3d(grid_ref.clone(), options.clone());

        output
            .into_data()
            .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
    }
}

/// The trilinear kernel (and the nearest fallback) agree with the reference backend for every
/// padding mode and `align_corners` setting, on coordinates reaching past both faces of every axis.
#[test]
pub fn grid_sample_3d_should_match_reference_backend() {
    let device = Default::default();
    let tensor = TestTensor::<5>::random([2, 3, 5, 6, 7], Distribution::Default, &device);
    let grid = TestTensor::<5>::random([2, 4, 5, 6, 3], Distribution::Uniform(-1.3, 1.3), &device);

    assert_matches_reference(tensor, grid);
}

/// A grid produced by permuting a component-first `[N, 3, D, H, W]` tensor keeps a non-unit
/// stride on its last axis; the kernel must honour it rather than assume contiguity.
#[test]
pub fn grid_sample_3d_should_match_reference_backend_permuted_grid() {
    let device = Default::default();
    let tensor = TestTensor::<5>::random([2, 2, 4, 5, 6], Distribution::Default, &device);
    let grid = TestTensor::<5>::random([2, 3, 3, 4, 5], Distribution::Uniform(-1.3, 1.3), &device)
        .permute([0, 2, 3, 4, 1]);

    assert_matches_reference(tensor, grid);
}

/// One thread handles all channels of a spatial position, so a channel count larger than the
/// spatial extent exercises the channel loop.
#[test]
pub fn grid_sample_3d_should_match_reference_backend_many_channels() {
    let device = Default::default();
    let tensor = TestTensor::<5>::random([1, 32, 2, 3, 3], Distribution::Default, &device);
    let grid = TestTensor::<5>::random([1, 2, 2, 2, 3], Distribution::Uniform(-1.0, 1.0), &device);

    assert_matches_reference(tensor, grid);
}
