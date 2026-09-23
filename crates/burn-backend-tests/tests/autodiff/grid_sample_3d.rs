use super::*;
use burn_tensor::{
    TensorData, Tolerance,
    ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode},
};

/// The forward sums eight weighted corners and the backward scatters them back, both in the
/// element type, so half precision loses a few units in the third decimal.
fn tolerance() -> Tolerance<FloatElem> {
    Tolerance::default()
        .set_half_precision_relative(5e-2)
        .set_half_precision_absolute(1e-2)
}

// Expected values in this file were computed with PyTorch in float64:
//
// ```python
// import torch, torch.nn.functional as F
// i = torch.arange(36, dtype=torch.float64)
// vol = ((i * 7) % 11 - 5).reshape(1, 2, 2, 3, 3).requires_grad_(True)
// grid = torch.tensor(GRID, dtype=torch.float64).requires_grad_(True)  # [1, 1, 2, 2, 3]
// w = (0.5 + 0.25 * torch.arange(8, dtype=torch.float64)).reshape(1, 2, 1, 2, 2)
// out = F.grid_sample(vol, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
// (out * w).sum().backward()
// vol.grad, grid.grad
// ```

/// `[1, 2, 2, 3, 3]` volume holding `(7 i mod 11) - 5` at flat index `i`.
fn volume() -> TestTensor<5> {
    TestTensor::<5>::from_data(
        [[
            [
                [[-5.0, 2.0, -2.0], [5.0, 1.0, -3.0], [4.0, 0.0, -4.0]],
                [[3.0, -1.0, -5.0], [2.0, -2.0, 5.0], [1.0, -3.0, 4.0]],
            ],
            [
                [[0.0, -4.0, 3.0], [-1.0, -5.0, 2.0], [-2.0, 5.0, 1.0]],
                [[-3.0, 4.0, 0.0], [-4.0, 3.0, -1.0], [-5.0, 2.0, -2.0]],
            ],
        ]],
        &AutodiffDevice::new(),
    )
}

/// Four `(x, y, z)` sampling locations as a `[1, 1, 2, 2, 3]` grid: an interior point, a point
/// on the `x = -1` face, a point past the `x = 1` face and a point outside on `y` and `z`.
fn grid() -> TestTensor<5> {
    TestTensor::<5>::from_data(
        [[[
            [[0.3, -0.2, 0.1], [-1.0, 0.5, 0.0]],
            [[1.4, -0.7, 0.3], [0.0, 2.2, -1.5]],
        ]]],
        &AutodiffDevice::new(),
    )
}

/// Per-element weights on the output, so every `grad_output` entry is distinct.
fn weights() -> TestTensor<5> {
    TestTensor::<5>::from_data(
        [[[[[0.5, 0.75], [1.0, 1.25]]], [[[1.5, 1.75], [2.0, 2.25]]]]],
        &AutodiffDevice::new(),
    )
}

fn make_options(
    mode: InterpolateMode,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> GridSampleOptions {
    GridSampleOptions::new(mode)
        .with_padding_mode(padding_mode)
        .with_align_corners(align_corners)
}

/// Runs the forward and backward, returning `(output, grad_volume, grad_grid)`.
fn run(options: GridSampleOptions) -> (TensorData, TensorData, TensorData) {
    let volume = volume().require_grad();
    let grid = grid().require_grad();

    let output = volume.clone().grid_sample_3d(grid.clone(), options);
    let grads = output.clone().mul(weights()).sum().backward();

    (
        output.into_data(),
        volume.grad(&grads).unwrap().into_data(),
        grid.grad(&grads).unwrap().into_data(),
    )
}

fn assert_case(
    options: GridSampleOptions,
    expected_output: TensorData,
    expected_grad_volume: TensorData,
    expected_grad_grid: TensorData,
) {
    let (output, grad_volume, grad_grid) = run(options);
    output.assert_approx_eq::<FloatElem>(&expected_output, tolerance());
    grad_volume.assert_approx_eq::<FloatElem>(&expected_grad_volume, tolerance());
    grad_grid.assert_approx_eq::<FloatElem>(&expected_grad_grid, tolerance());
}

#[test]
fn test_grid_sample_3d_trilinear_zeros() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='zeros', align_corners=False)
    assert_case(
        GridSampleOptions::default(),
        TensorData::from([[
            [[[-0.221, 1.375], [0.0, 0.0]]],
            [[[0.28, -1.625], [0.0, 0.0]]],
        ]]),
        TensorData::from([[
            [
                [
                    [0.0, 0.033, 0.027],
                    [0.046875, 0.077, 0.063],
                    [0.140625, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0495, 0.0405],
                    [0.046875, 0.1155, 0.0945],
                    [0.140625, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.099, 0.081],
                    [0.109375, 0.231, 0.189],
                    [0.328125, 0.0, 0.0],
                ],
                [
                    [0.0, 0.1485, 0.1215],
                    [0.109375, 0.3465, 0.2835],
                    [0.328125, 0.0, 0.0],
                ],
            ],
        ]]),
        TensorData::from([[[
            [[1.365, -0.7725, 4.8075], [-5.4375, -1.875, -3.75]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]]]),
    );
}

#[test]
fn test_grid_sample_3d_trilinear_zeros_align_corners() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='zeros', align_corners=True)
    assert_case(
        make_options(
            InterpolateMode::Bilinear,
            GridSamplePaddingMode::Zeros,
            true,
        ),
        TensorData::from([[
            [[[-0.198, 3.0], [-1.263, 0.0]]],
            [[[-0.115, -3.0], [0.45, 0.0]]],
        ]]),
        TensorData::from([[
            [
                [
                    [0.0, 0.0315, 0.1605],
                    [0.1875, 0.126, 0.117],
                    [0.1875, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0385, 0.2895],
                    [0.1875, 0.154, 0.183],
                    [0.1875, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.0945, 0.3345],
                    [0.4375, 0.378, 0.288],
                    [0.4375, 0.0, 0.0],
                ],
                [
                    [0.0, 0.1155, 0.5955],
                    [0.4375, 0.462, 0.432],
                    [0.4375, 0.0, 0.0],
                ],
            ],
        ]]),
        TensorData::from([[[
            [[1.845, -1.0925, 3.435], [4.4375, -2.5, -3.75]],
            [[0.605, 2.49, -1.71], [0.0, 0.0, 0.0]],
        ]]]),
    );
}

/// Border padding: a coordinate clamped onto a face receives no gradient along that axis.
#[test]
fn test_grid_sample_3d_trilinear_border() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='border', align_corners=False)
    assert_case(
        make_options(
            InterpolateMode::Bilinear,
            GridSamplePaddingMode::Border,
            false,
        ),
        TensorData::from([[
            [[[-0.221, 2.75], [-4.4, 0.0]]],
            [[[0.28, -3.25], [0.6, 5.0]]],
        ]]),
        TensorData::from([[
            [
                [
                    [0.0, 0.033, 0.227],
                    [0.09375, 0.077, 0.063],
                    [0.28125, 1.25, 0.0],
                ],
                [
                    [0.0, 0.0495, 0.8405],
                    [0.09375, 0.1155, 0.0945],
                    [0.28125, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.099, 0.481],
                    [0.21875, 0.231, 0.189],
                    [0.65625, 2.25, 0.0],
                ],
                [
                    [0.0, 0.1485, 1.7215],
                    [0.21875, 0.3465, 0.2835],
                    [0.65625, 0.0, 0.0],
                ],
            ],
        ]]),
        TensorData::from([[[
            [[1.365, -0.7725, 4.8075], [0.0, -3.75, -7.5]],
            [[0.0, 0.0, -9.0], [-21.0, 0.0, 0.0]],
        ]]]),
    );
}

#[test]
fn test_grid_sample_3d_trilinear_border_align_corners() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='border', align_corners=True)
    assert_case(
        make_options(
            InterpolateMode::Bilinear,
            GridSamplePaddingMode::Border,
            true,
        ),
        TensorData::from([[
            [[[-0.198, 3.0], [-2.105, 0.0]]],
            [[[-0.115, -3.0], [0.75, 5.0]]],
        ]]),
        TensorData::from([[
            [
                [
                    [0.0, 0.0315, 0.2585],
                    [0.1875, 0.126, 0.159],
                    [0.1875, 1.25, 0.0],
                ],
                [
                    [0.0, 0.0385, 0.4715],
                    [0.1875, 0.154, 0.261],
                    [0.1875, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.0945, 0.5305],
                    [0.4375, 0.378, 0.372],
                    [0.4375, 2.25, 0.0],
                ],
                [
                    [0.0, 0.1155, 0.9595],
                    [0.4375, 0.462, 0.588],
                    [0.4375, 0.0, 0.0],
                ],
            ],
        ]]),
        TensorData::from([[[
            [[1.845, -1.0925, 3.435], [0.0, -2.5, -3.75]],
            [[0.0, 4.15, -2.85], [-14.0, 0.0, 0.0]],
        ]]]),
    );
}

/// Reflection padding: the gradient flips sign at every fold of the triangle wave.
#[test]
fn test_grid_sample_3d_trilinear_reflection() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='reflection', align_corners=False)
    assert_case(
        make_options(
            InterpolateMode::Bilinear,
            GridSamplePaddingMode::Reflection,
            false,
        ),
        TensorData::from([[
            [[[-0.221, 2.75], [-4.0, 1.3]]],
            [[[0.28, -3.25], [0.78, -4.7]]],
        ]]),
        TensorData::from([[
            [
                [
                    [0.0, 0.428, 0.207],
                    [0.09375, 0.952, 0.063],
                    [0.28125, 0.0, 0.0],
                ],
                [
                    [0.0, 0.1295, 0.7605],
                    [0.09375, 0.1155, 0.0945],
                    [0.28125, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.814, 0.441],
                    [0.21875, 1.806, 0.189],
                    [0.65625, 0.0, 0.0],
                ],
                [
                    [0.0, 0.3085, 1.5615],
                    [0.21875, 0.3465, 0.2835],
                    [0.65625, 0.0, 0.0],
                ],
            ],
        ]]),
        TensorData::from([[[
            [[1.365, -0.7725, 4.8075], [0.0, -3.75, -7.5]],
            [[11.4, 0.0, -6.8], [16.125, 5.25, 0.0]],
        ]]]),
    );
}

#[test]
fn test_grid_sample_3d_trilinear_reflection_align_corners() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='reflection', align_corners=True)
    assert_case(
        make_options(
            InterpolateMode::Bilinear,
            GridSamplePaddingMode::Reflection,
            true,
        ),
        TensorData::from([[
            [[[-0.198, 3.0], [-1.363, 0.45]]],
            [[[-0.115, -3.0], [0.81, -2.8]]],
        ]]),
        TensorData::from([[
            [
                [
                    [0.0, 0.317, 0.1605],
                    [0.1875, 0.918, 0.117],
                    [0.1875, 0.0, 0.0],
                ],
                [
                    [0.0, 0.283, 0.2895],
                    [0.1875, 0.482, 0.183],
                    [0.1875, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.628, 0.3345],
                    [0.4375, 1.812, 0.288],
                    [0.4375, 0.0, 0.0],
                ],
                [
                    [0.0, 0.592, 0.5955],
                    [0.4375, 1.068, 0.432],
                    [0.4375, 0.0, 0.0],
                ],
            ],
        ]]),
        TensorData::from([[[
            [[1.845, -1.0925, 3.435], [0.0, -2.5, -3.75]],
            [[2.155, 1.29, 0.89], [7.3125, 3.5, -7.125]],
        ]]]),
    );
}

/// Nearest mode is piecewise constant in the grid: the sampled voxel receives the full
/// gradient and the grid receives none.
#[test]
fn test_grid_sample_3d_nearest() {
    // PyTorch grid_sample(mode='nearest', padding_mode='zeros', align_corners=False)
    assert_case(
        make_options(
            InterpolateMode::Nearest,
            GridSamplePaddingMode::Zeros,
            false,
        ),
        TensorData::from([[[[[-2.0, 4.0], [0.0, 0.0]]], [[[3.0, -2.0], [0.0, 0.0]]]]]),
        TensorData::from([[
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.75, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.0]],
            ],
        ]]),
        TensorData::zeros::<FloatElem, _>([1, 1, 2, 2, 3]),
    );
}

/// Only the sampled volume is tracked: its gradient must not depend on the grid being tracked.
#[test]
fn test_grid_sample_3d_volume_only_tracked() {
    let volume = volume().require_grad();
    let grid = grid();

    let output = volume
        .clone()
        .grid_sample_3d(grid, GridSampleOptions::default());
    let grads = output.mul(weights()).sum().backward();

    // Same gradient as `test_grid_sample_3d_trilinear_zeros`.
    let expected = TensorData::from([[
        [
            [
                [0.0, 0.033, 0.027],
                [0.046875, 0.077, 0.063],
                [0.140625, 0.0, 0.0],
            ],
            [
                [0.0, 0.0495, 0.0405],
                [0.046875, 0.1155, 0.0945],
                [0.140625, 0.0, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.099, 0.081],
                [0.109375, 0.231, 0.189],
                [0.328125, 0.0, 0.0],
            ],
            [
                [0.0, 0.1485, 0.1215],
                [0.109375, 0.3465, 0.2835],
                [0.328125, 0.0, 0.0],
            ],
        ],
    ]]);
    volume
        .grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance());
}

/// Only the grid is tracked: the volume is checkpointed for the grid gradient even though it
/// has no gradient of its own.
#[test]
fn test_grid_sample_3d_grid_only_tracked() {
    let volume = volume();
    let grid = grid().require_grad();

    let output = volume.grid_sample_3d(grid.clone(), GridSampleOptions::default());
    let grads = output.mul(weights()).sum().backward();

    // Same gradient as `test_grid_sample_3d_trilinear_zeros`.
    let expected = TensorData::from([[[
        [[1.365, -0.7725, 4.8075], [-5.4375, -1.875, -3.75]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]]]);
    grid.grad(&grads)
        .unwrap()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance());
}
