use super::*;
use burn_backend::ops::grid_sample::float_grid_sample_3d_ref;
use burn_dispatch::Dispatch;
use burn_tensor::{
    TensorData, Tolerance,
    ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode},
};

/// Eight weighted corner values are summed per output, in the element type on backends that
/// sample natively, so half precision loses a few units in the third decimal.
fn tolerance() -> Tolerance<FloatElem> {
    Tolerance::default()
        .set_half_precision_relative(5e-2)
        .set_half_precision_absolute(1e-2)
}

// Expected values in this file were computed with PyTorch in float64:
//
// ```python
// import torch, torch.nn.functional as F
// vol = torch.arange(1, 25, dtype=torch.float64).reshape(1, 1, 2, 3, 4)
// grid = torch.tensor(GRID, dtype=torch.float64)  # [N, D_out, H_out, W_out, 3], (x, y, z)
// F.grid_sample(vol, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
// ```

/// A `[1, 1, 2, 3, 4]` volume holding `12 z + 4 y + x + 1`, so every voxel is distinct and a
/// swapped axis or a wrong stride shows up as a value mismatch.
fn volume() -> TestTensor<5> {
    TestTensor::<5>::from_data(
        [[[
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]]],
        &Default::default(),
    )
}

/// Six sampling locations, `(x, y, z)` each, as a `[1, 1, 2, 3, 3]` grid:
/// an interior point, the front-top-left corner (a nearest-neighbour tie when
/// `align_corners = false`), one point outside each `x` face, a point two folds outside the
/// `y` face and a point outside on two axes at once.
fn grid() -> TestTensor<5> {
    TestTensor::<5>::from_data(
        [[[
            [[0.25, -0.5, 0.1], [-1.0, -1.0, -1.0], [-1.5, 0.2, 0.0]],
            [[1.75, 0.0, 0.4], [0.0, -2.5, 0.0], [1.3, 0.6, -1.7]],
        ]]],
        &Default::default(),
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

fn assert_sample(
    tensor: TestTensor<5>,
    grid: TestTensor<5>,
    options: GridSampleOptions,
    expected: TensorData,
) {
    let output = tensor.grid_sample_3d(grid, options);
    output
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance());
}

/// Default options: trilinear, zeros padding, `align_corners = false`.
#[test]
fn should_grid_sample_3d_default() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='zeros', align_corners=False)
    let expected = TensorData::from([[[[[11.2, 0.125, 0.0], [0.0, 0.0, 0.0]]]]]);
    assert_sample(volume(), grid(), GridSampleOptions::default(), expected);
}

#[test]
fn should_grid_sample_3d_zeros_align_corners() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='zeros', align_corners=True)
    let expected = TensorData::from([[[[[11.475, 1.0, 2.95], [0.0, 0.0, 3.718]]]]]);
    let options = make_options(
        InterpolateMode::Bilinear,
        GridSamplePaddingMode::Zeros,
        true,
    );
    assert_sample(volume(), grid(), options, expected);
}

#[test]
fn should_grid_sample_3d_border() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='border', align_corners=False)
    let expected = TensorData::from([[[[[11.2, 1.0, 12.2], [18.8, 8.5, 11.6]]]]]);
    let options = make_options(
        InterpolateMode::Bilinear,
        GridSamplePaddingMode::Border,
        false,
    );
    assert_sample(volume(), grid(), options, expected);
}

#[test]
fn should_grid_sample_3d_border_align_corners() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='border', align_corners=True)
    let expected = TensorData::from([[[[[11.475, 1.0, 11.8], [16.4, 8.5, 10.4]]]]]);
    let options = make_options(
        InterpolateMode::Bilinear,
        GridSamplePaddingMode::Border,
        true,
    );
    assert_sample(volume(), grid(), options, expected);
}

#[test]
fn should_grid_sample_3d_reflection() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='reflection', align_corners=False)
    let expected = TensorData::from([[[[[11.2, 1.0, 12.7], [17.8, 15.5, 13.9]]]]]);
    let options = make_options(
        InterpolateMode::Bilinear,
        GridSamplePaddingMode::Reflection,
        false,
    );
    assert_sample(volume(), grid(), options, expected);
}

#[test]
fn should_grid_sample_3d_reflection_align_corners() {
    // PyTorch grid_sample(mode='bilinear', padding_mode='reflection', align_corners=True)
    let expected = TensorData::from([[[[[11.475, 1.0, 12.55], [15.275, 14.5, 14.15]]]]]);
    let options = make_options(
        InterpolateMode::Bilinear,
        GridSamplePaddingMode::Reflection,
        true,
    );
    assert_sample(volume(), grid(), options, expected);
}

/// Nearest mode. The corner sample lands exactly on `-0.5` voxel coordinates with
/// `align_corners = false`, so it also pins the round-half-to-even tie rule.
#[test]
fn should_grid_sample_3d_nearest_zeros() {
    // PyTorch grid_sample(mode='nearest', padding_mode='zeros', align_corners=False)
    let expected = TensorData::from([[[[[15.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]]);
    let options = make_options(
        InterpolateMode::Nearest,
        GridSamplePaddingMode::Zeros,
        false,
    );
    assert_sample(volume(), grid(), options, expected);

    // PyTorch grid_sample(mode='nearest', padding_mode='zeros', align_corners=True)
    let expected = TensorData::from([[[[[15.0, 1.0, 0.0], [0.0, 0.0, 12.0]]]]]);
    let options = make_options(InterpolateMode::Nearest, GridSamplePaddingMode::Zeros, true);
    assert_sample(volume(), grid(), options, expected);
}

#[test]
fn should_grid_sample_3d_nearest_border() {
    // PyTorch grid_sample(mode='nearest', padding_mode='border', align_corners=False/True)
    let expected = TensorData::from([[[[[15.0, 1.0, 5.0], [20.0, 3.0, 12.0]]]]]);
    for align_corners in [false, true] {
        let options = make_options(
            InterpolateMode::Nearest,
            GridSamplePaddingMode::Border,
            align_corners,
        );
        assert_sample(volume(), grid(), options, expected.clone());
    }
}

#[test]
fn should_grid_sample_3d_nearest_reflection() {
    // PyTorch grid_sample(mode='nearest', padding_mode='reflection', align_corners=False)
    let expected = TensorData::from([[[[[15.0, 1.0, 5.0], [19.0, 11.0, 12.0]]]]]);
    let options = make_options(
        InterpolateMode::Nearest,
        GridSamplePaddingMode::Reflection,
        false,
    );
    assert_sample(volume(), grid(), options, expected);

    // PyTorch grid_sample(mode='nearest', padding_mode='reflection', align_corners=True)
    let expected = TensorData::from([[[[[15.0, 1.0, 6.0], [19.0, 11.0, 12.0]]]]]);
    let options = make_options(
        InterpolateMode::Nearest,
        GridSamplePaddingMode::Reflection,
        true,
    );
    assert_sample(volume(), grid(), options, expected);
}

/// Several batches and channels: the grid is shared across channels but not across batches.
///
/// The volume holds `(7 i mod 11) - 5` at flat index `i`, and the grid is `[2, 1, 2, 2, 3]`.
#[test]
fn should_grid_sample_3d_batched_multichannel() {
    let device = Default::default();
    let tensor = TestTensor::<5>::from_data(
        [
            [
                [
                    [[-5.0, 2.0, -2.0], [5.0, 1.0, -3.0]],
                    [[4.0, 0.0, -4.0], [3.0, -1.0, -5.0]],
                ],
                [
                    [[2.0, -2.0, 5.0], [1.0, -3.0, 4.0]],
                    [[0.0, -4.0, 3.0], [-1.0, -5.0, 2.0]],
                ],
            ],
            [
                [
                    [[-2.0, 5.0, 1.0], [-3.0, 4.0, 0.0]],
                    [[-4.0, 3.0, -1.0], [-5.0, 2.0, -2.0]],
                ],
                [
                    [[5.0, 1.0, -3.0], [4.0, 0.0, -4.0]],
                    [[3.0, -1.0, -5.0], [2.0, -2.0, 5.0]],
                ],
            ],
        ],
        &device,
    );
    let grid = TestTensor::<5>::from_data(
        [
            [[
                [[-0.6, 0.3, 0.2], [0.9, -0.4, -0.8]],
                [[0.1, 0.7, 0.5], [-1.1, 1.2, 0.0]],
            ]],
            [[
                [[0.4, -0.9, 0.6], [-0.2, 0.2, -0.3]],
                [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            ]],
        ],
        &device,
    );

    // PyTorch grid_sample(mode='bilinear', padding_mode='zeros', align_corners=False)
    let expected = TensorData::from([
        [
            [[[2.806, -0.9555], [-1.28, 0.42]]],
            [[[-0.6, 2.2295], [-3.16, 0.0]]],
        ],
        [
            [[[0.324, 1.8], [-0.25, 3.5]]],
            [[[-1.836, 1.1], [0.625, -0.5]]],
        ],
    ]);
    assert_sample(
        tensor.clone(),
        grid.clone(),
        GridSampleOptions::default(),
        expected,
    );

    // PyTorch grid_sample(mode='bilinear', padding_mode='border', align_corners=True)
    let expected = TensorData::from([
        [
            [[[1.626, -2.1], [-0.75, 4.0]]],
            [[[-1.45, 3.8], [-3.65, 0.0]]],
        ],
        [[[[1.75, 2.3], [-2.0, 3.5]]], [[[-2.074, 0.5], [5.0, -0.5]]]],
    ]);
    let options = make_options(
        InterpolateMode::Bilinear,
        GridSamplePaddingMode::Border,
        true,
    );
    assert_sample(tensor.clone(), grid.clone(), options, expected);

    // PyTorch grid_sample(mode='nearest', padding_mode='reflection', align_corners=False)
    let expected = TensorData::from([
        [[[[3.0, -2.0], [-1.0, 5.0]]], [[[-1.0, 5.0], [-5.0, 1.0]]]],
        [[[[-1.0, 4.0], [-2.0, 5.0]]], [[[-5.0, 0.0], [5.0, 1.0]]]],
    ]);
    let options = make_options(
        InterpolateMode::Nearest,
        GridSamplePaddingMode::Reflection,
        false,
    );
    assert_sample(tensor, grid, options, expected);
}

/// A single-voxel depth axis: with `align_corners = true` the unnormalization divides by
/// `size - 1 = 0`, and every reflection span collapses to a point.
#[test]
fn should_grid_sample_3d_degenerate_depth() {
    let device = Default::default();
    let tensor = TestTensor::<5>::from_data([[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]], &device);
    let grid = TestTensor::<5>::from_data([[[[[0.2, -0.3, 0.4], [-1.6, 1.3, -0.9]]]]], &device);

    // PyTorch grid_sample(mode='bilinear', padding_mode=..., align_corners=...)
    let cases = [
        (GridSamplePaddingMode::Zeros, false, [2.32, 0.0]),
        (GridSamplePaddingMode::Zeros, true, [3.25, 1.36]),
        (GridSamplePaddingMode::Border, false, [2.9, 4.0]),
        (GridSamplePaddingMode::Border, true, [3.25, 4.0]),
        (GridSamplePaddingMode::Reflection, false, [2.9, 4.4]),
        (GridSamplePaddingMode::Reflection, true, [3.25, 4.15]),
    ];
    for (padding_mode, align_corners, expected) in cases {
        let options = make_options(InterpolateMode::Bilinear, padding_mode, align_corners);
        assert_sample(
            tensor.clone(),
            grid.clone(),
            options,
            TensorData::from([[[[expected]]]]),
        );
    }

    // PyTorch grid_sample(mode='nearest', padding_mode=..., align_corners=...)
    let cases = [
        (GridSamplePaddingMode::Zeros, false, [2.0, 0.0]),
        (GridSamplePaddingMode::Zeros, true, [2.0, 0.0]),
        (GridSamplePaddingMode::Border, false, [2.0, 4.0]),
        (GridSamplePaddingMode::Border, true, [2.0, 4.0]),
        (GridSamplePaddingMode::Reflection, false, [2.0, 4.0]),
        (GridSamplePaddingMode::Reflection, true, [2.0, 5.0]),
    ];
    for (padding_mode, align_corners, expected) in cases {
        let options = make_options(InterpolateMode::Nearest, padding_mode, align_corners);
        assert_sample(
            tensor.clone(),
            grid.clone(),
            options,
            TensorData::from([[[[expected]]]]),
        );
    }
}

/// A grid held component-first, `[N, 3, D, H, W]`, and moved to the trailing position with a
/// `permute` view has the required shape but a non-unit component stride. It must sample the
/// same points as the contiguous grid.
#[test]
fn should_grid_sample_3d_permuted_grid_view() {
    let device = Default::default();
    // The same six points as `grid()`, stored as all x, then all y, then all z.
    let component_major = TestTensor::<5>::from_data(
        [[
            [[[0.25, -1.0, -1.5], [1.75, 0.0, 1.3]]],
            [[[-0.5, -1.0, 0.2], [0.0, -2.5, 0.6]]],
            [[[0.1, -1.0, 0.0], [0.4, 0.0, -1.7]]],
        ]],
        &device,
    );
    let permuted = component_major.permute([0, 2, 3, 4, 1]);
    assert_eq!(permuted.dims(), [1, 1, 2, 3, 3]);

    let options = make_options(
        InterpolateMode::Bilinear,
        GridSamplePaddingMode::Reflection,
        false,
    );
    let expected = volume().grid_sample_3d(grid(), options.clone()).to_data();
    let output = volume().grid_sample_3d(permuted, options);
    output
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance());
}

/// The output is allocated over the input's batch, so a grid with a different batch extent is
/// rejected up front rather than read out of bounds or silently truncated.
#[test]
#[should_panic(expected = "=== Tensor Operation Error ===")]
fn should_grid_sample_3d_reject_batch_mismatch() {
    let device = Default::default();
    let tensor = TestTensor::<5>::zeros([2, 1, 2, 2, 2], &device);
    let grid = TestTensor::<5>::zeros([1, 1, 2, 2, 3], &device);

    let _ = tensor.grid_sample_3d(grid, GridSampleOptions::default());
}

/// The grid's last dimension must hold exactly the three `(x, y, z)` coordinates.
#[test]
#[should_panic(expected = "=== Tensor Operation Error ===")]
fn should_grid_sample_3d_reject_wrong_coordinate_count() {
    let device = Default::default();
    let tensor = TestTensor::<5>::zeros([1, 1, 2, 2, 2], &device);
    let grid = TestTensor::<5>::zeros([1, 1, 2, 2, 2], &device);

    let _ = tensor.grid_sample_3d(grid, GridSampleOptions::default());
}

/// Both tensors must be rank 5.
#[test]
#[should_panic(expected = "=== Tensor Operation Error ===")]
fn should_grid_sample_3d_reject_wrong_rank() {
    let device = Default::default();
    let tensor = TestTensor::<4>::zeros([1, 1, 2, 2], &device);
    let grid = TestTensor::<4>::zeros([1, 2, 2, 3], &device);

    let _ = tensor.grid_sample_3d(grid, GridSampleOptions::default());
}

/// Every combination of interpolation mode, padding mode and `align_corners` on both fixtures.
fn all_options() -> Vec<GridSampleOptions> {
    let mut options = Vec::new();
    for mode in [InterpolateMode::Bilinear, InterpolateMode::Nearest] {
        for padding_mode in [
            GridSamplePaddingMode::Zeros,
            GridSamplePaddingMode::Border,
            GridSamplePaddingMode::Reflection,
        ] {
            for align_corners in [false, true] {
                options.push(make_options(mode.clone(), padding_mode, align_corners));
            }
        }
    }
    options
}

/// The reference implementation, which decomposes into primitive tensor ops and is what a
/// backend without a native sampler runs, agrees with the backend under test.
///
/// The decomposition fuses, on CubeCL backends, into a kernel that combines two masked,
/// broadcast-weighted terms and writes its output in place over one of them. On the WGSL path
/// the NVIDIA driver miscompiles that kernel: the last lane of every vectorized mask reads as
/// `true`. The same kernel is correct on Mesa's lavapipe, through cubecl's SPIR-V compiler on
/// the same GPU, and on the CPU backend, so the comparison is skipped only where the WGSL path
/// is the one under test.
#[test]
#[cfg_attr(
    all(feature = "wgpu", not(feature = "vulkan"), not(feature = "metal")),
    ignore = "the NVIDIA driver miscompiles the fused masked-broadcast kernel on the WGSL path"
)]
fn should_grid_sample_3d_reference_match_native() {
    let device = Default::default();
    let fixtures = [
        (volume(), grid()),
        (
            TestTensor::<5>::from_data(
                [
                    [
                        [
                            [[-5.0, 2.0, -2.0], [5.0, 1.0, -3.0]],
                            [[4.0, 0.0, -4.0], [3.0, -1.0, -5.0]],
                        ],
                        [
                            [[2.0, -2.0, 5.0], [1.0, -3.0, 4.0]],
                            [[0.0, -4.0, 3.0], [-1.0, -5.0, 2.0]],
                        ],
                    ],
                    [
                        [
                            [[-2.0, 5.0, 1.0], [-3.0, 4.0, 0.0]],
                            [[-4.0, 3.0, -1.0], [-5.0, 2.0, -2.0]],
                        ],
                        [
                            [[5.0, 1.0, -3.0], [4.0, 0.0, -4.0]],
                            [[3.0, -1.0, -5.0], [2.0, -2.0, 5.0]],
                        ],
                    ],
                ],
                &device,
            ),
            TestTensor::<5>::from_data(
                [
                    [[
                        [[-0.6, 0.3, 0.2], [0.9, -0.4, -0.8]],
                        [[0.1, 0.7, 0.5], [-1.1, 1.2, 0.0]],
                    ]],
                    [[
                        [[0.4, -0.9, 0.6], [-0.2, 0.2, -0.3]],
                        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                    ]],
                ],
                &device,
            ),
        ),
    ];

    for (tensor, grid) in fixtures {
        for options in all_options() {
            let expected = tensor
                .clone()
                .grid_sample_3d(grid.clone(), options.clone())
                .into_data();

            let reference = TestTensor::<5>::from_dispatch(float_grid_sample_3d_ref::<Dispatch>(
                tensor.clone().into_dispatch(),
                grid.clone().into_dispatch(),
                options.clone(),
            ));

            reference
                .into_data()
                .assert_approx_eq::<FloatElem>(&expected, tolerance());
        }
    }
}
