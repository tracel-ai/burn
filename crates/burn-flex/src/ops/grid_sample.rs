//! Grid sampling operations for FlexTensor.
//!
//! Supported dtypes: f32, f64, f16, bf16. All dtypes share a single f64
//! compute path. This is required for f16/bf16 correctness (so coordinate
//! math, bilinear weights, and accumulated samples keep full precision) and
//! incidentally gives f32 a small accuracy bump at the cost of extra casts.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::element::cast::ToElement;
use burn_backend::ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode};
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};

use num_traits::{Float, NumCast};

use crate::{FlexTensor, Layout};

/// Grid sample 2D (bilinear and nearest-neighbor interpolation).
///
/// Input tensor shape: [N, C, H_in, W_in]
/// Grid shape: [N, H_out, W_out, 2] (x, y normalized to [-1, 1])
/// Output shape: [N, C, H_out, W_out]
pub fn grid_sample_2d(
    tensor: FlexTensor,
    grid: FlexTensor,
    options: GridSampleOptions,
) -> FlexTensor {
    match options.mode {
        InterpolateMode::Bilinear | InterpolateMode::Nearest => {}
        other => panic!("grid_sample_2d: {:?} mode is not supported", other),
    }

    let tensor = tensor.to_contiguous();
    let grid = grid.to_contiguous();

    match tensor.dtype() {
        DType::F32 => grid_sample_2d_impl::<f32>(tensor, grid, options),
        DType::F64 => grid_sample_2d_impl::<f64>(tensor, grid, options),
        DType::F16 => grid_sample_2d_impl::<f16>(tensor, grid, options),
        DType::BF16 => grid_sample_2d_impl::<bf16>(tensor, grid, options),
        _ => panic!("grid_sample_2d: unsupported dtype {:?}", tensor.dtype()),
    }
}

fn grid_sample_2d_impl<T>(
    tensor: FlexTensor,
    grid: FlexTensor,
    options: GridSampleOptions,
) -> FlexTensor
where
    T: Float + Element + bytemuck::Pod,
{
    let t_shape = tensor.layout().shape();
    let g_shape = grid.layout().shape();

    assert_eq!(t_shape.num_dims(), 4, "grid_sample_2d: input must be 4D");
    assert_eq!(g_shape.num_dims(), 4, "grid_sample_2d: grid must be 4D");
    assert_eq!(g_shape[3], 2, "grid_sample_2d: grid last dim must be 2");
    assert_eq!(
        t_shape[0], g_shape[0],
        "grid_sample_2d: batch size mismatch"
    );

    let batch_size = t_shape[0];
    let channels = t_shape[1];
    let h_in = t_shape[2];
    let w_in = t_shape[3];
    let h_out = g_shape[1];
    let w_out = g_shape[2];

    let tensor_data: &[T] = tensor.storage();
    let grid_data: &[T] = grid.storage();

    let out_shape = Shape::from(vec![batch_size, channels, h_out, w_out]);
    let out_len = batch_size * channels * h_out * w_out;
    let mut output: Vec<T> = vec![T::zero(); out_len];

    let align = options.align_corners;
    let pad_mode = options.padding_mode;

    let t_stride_n = channels * h_in * w_in;
    let t_stride_c = h_in * w_in;
    let t_stride_h = w_in;

    let g_stride_n = h_out * w_out * 2;
    let g_stride_h = w_out * 2;

    let o_stride_n = channels * h_out * w_out;
    let o_stride_c = h_out * w_out;
    let o_stride_h = w_out;

    // Low-precision types (f16/bf16) are widened to f64 for all arithmetic so
    // that coordinate math, weights, and accumulated samples keep full precision.
    //
    // The from_f64 unwrap is unreachable for any well-formed input: bilinear
    // is a convex combination of finite samples so the result stays bounded by
    // the sample envelope. The `half` crate's `NumCast` impl for f16/bf16
    // forwards through `to_f32` and maps non-finite inputs to `Some(inf/nan)`;
    // `num_traits`'s f32/f64 impls do the same. The message-bearing panic is a
    // diagnostic hatch for future dtypes where this invariant does not hold.
    let to_f64 = |x: T| -> f64 { ToElement::to_f64(&x) };
    let from_f64 = |x: f64| -> T {
        <T as NumCast>::from(x).unwrap_or_else(|| {
            panic!(
                "grid_sample_2d: NumCast::from({x:?}) to {:?} returned None",
                T::dtype()
            )
        })
    };

    for b in 0..batch_size {
        for y in 0..h_out {
            for x in 0..w_out {
                let g_idx = b * g_stride_n + y * g_stride_h + x * 2;
                let sample_x = to_f64(grid_data[g_idx]);
                let sample_y = to_f64(grid_data[g_idx + 1]);

                let (px, py) = if align {
                    let px = (sample_x + 1.0) * ((w_in - 1) as f64) / 2.0;
                    let py = (sample_y + 1.0) * ((h_in - 1) as f64) / 2.0;
                    (px, py)
                } else {
                    let px = (sample_x + 1.0) * (w_in as f64) / 2.0 - 0.5;
                    let py = (sample_y + 1.0) * (h_in as f64) / 2.0 - 0.5;
                    (px, py)
                };

                let (px, py) = apply_padding(px, py, w_in, h_in, pad_mode, align);

                let read = |t_base: usize, xi: i64, yi: i64| -> f64 {
                    match pad_mode {
                        GridSamplePaddingMode::Zeros => {
                            if xi >= 0 && xi < w_in as i64 && yi >= 0 && yi < h_in as i64 {
                                to_f64(tensor_data[t_base + yi as usize * t_stride_h + xi as usize])
                            } else {
                                0.0
                            }
                        }
                        GridSamplePaddingMode::Border | GridSamplePaddingMode::Reflection => {
                            let xi = xi.clamp(0, (w_in - 1) as i64) as usize;
                            let yi = yi.clamp(0, (h_in - 1) as i64) as usize;
                            to_f64(tensor_data[t_base + yi * t_stride_h + xi])
                        }
                    }
                };

                for c in 0..channels {
                    let t_base = b * t_stride_n + c * t_stride_c;
                    let o_idx = b * o_stride_n + c * o_stride_c + y * o_stride_h + x;

                    let val = if matches!(options.mode, InterpolateMode::Nearest) {
                        // Ties round to even, like PyTorch's `nearbyint`.
                        let xi = libm::roundeven(px) as i64;
                        let yi = libm::roundeven(py) as i64;
                        read(t_base, xi, yi)
                    } else {
                        // Bilinear
                        let x0 = px.floor() as i64;
                        let y0 = py.floor() as i64;
                        let x1 = x0 + 1;
                        let y1 = y0 + 1;

                        let x_frac = px - px.floor();
                        let y_frac = py - py.floor();

                        let w00 = (1.0 - x_frac) * (1.0 - y_frac);
                        let w01 = (1.0 - x_frac) * y_frac;
                        let w10 = x_frac * (1.0 - y_frac);
                        let w11 = x_frac * y_frac;

                        read(t_base, x0, y0) * w00
                            + read(t_base, x0, y1) * w01
                            + read(t_base, x1, y0) * w10
                            + read(t_base, x1, y1) * w11
                    };

                    output[o_idx] = from_f64(val);
                }
            }
        }
    }

    let bytes = Bytes::from_elems(output);
    FlexTensor::new(bytes, Layout::contiguous(out_shape), T::dtype())
}

/// Grid sample 3D (trilinear and nearest-neighbor interpolation).
///
/// Input tensor shape: [N, C, D_in, H_in, W_in]
/// Grid shape: [N, D_out, H_out, W_out, 3] (x, y, z normalized to [-1, 1]; `x` indexes `W_in`,
/// `y` indexes `H_in` and `z` indexes `D_in`)
/// Output shape: [N, C, D_out, H_out, W_out]
///
/// `InterpolateMode::Bilinear` selects trilinear interpolation at this rank.
pub fn grid_sample_3d(
    tensor: FlexTensor,
    grid: FlexTensor,
    options: GridSampleOptions,
) -> FlexTensor {
    match options.mode {
        InterpolateMode::Bilinear | InterpolateMode::Nearest => {}
        other => panic!("grid_sample_3d: {:?} mode is not supported", other),
    }

    let tensor = tensor.to_contiguous();
    let grid = grid.to_contiguous();

    match tensor.dtype() {
        DType::F32 => grid_sample_3d_impl::<f32>(tensor, grid, options),
        DType::F64 => grid_sample_3d_impl::<f64>(tensor, grid, options),
        DType::F16 => grid_sample_3d_impl::<f16>(tensor, grid, options),
        DType::BF16 => grid_sample_3d_impl::<bf16>(tensor, grid, options),
        _ => panic!("grid_sample_3d: unsupported dtype {:?}", tensor.dtype()),
    }
}

fn grid_sample_3d_impl<T>(
    tensor: FlexTensor,
    grid: FlexTensor,
    options: GridSampleOptions,
) -> FlexTensor
where
    T: Float + Element + bytemuck::Pod,
{
    let t_shape = tensor.layout().shape();
    let g_shape = grid.layout().shape();

    assert_eq!(t_shape.num_dims(), 5, "grid_sample_3d: input must be 5D");
    assert_eq!(g_shape.num_dims(), 5, "grid_sample_3d: grid must be 5D");
    assert_eq!(g_shape[4], 3, "grid_sample_3d: grid last dim must be 3");
    assert_eq!(
        t_shape[0], g_shape[0],
        "grid_sample_3d: batch size mismatch"
    );

    let batch_size = t_shape[0];
    let channels = t_shape[1];
    let d_in = t_shape[2];
    let h_in = t_shape[3];
    let w_in = t_shape[4];
    let d_out = g_shape[1];
    let h_out = g_shape[2];
    let w_out = g_shape[3];

    let tensor_data: &[T] = tensor.storage();
    let grid_data: &[T] = grid.storage();

    let out_shape = Shape::from(vec![batch_size, channels, d_out, h_out, w_out]);
    let out_len = batch_size * channels * d_out * h_out * w_out;
    let mut output: Vec<T> = vec![T::zero(); out_len];

    let align = options.align_corners;
    let pad_mode = options.padding_mode;
    let size = [w_in, h_in, d_in];

    let t_stride_n = channels * d_in * h_in * w_in;
    let t_stride_c = d_in * h_in * w_in;
    let t_stride_d = h_in * w_in;
    let t_stride_h = w_in;

    let g_stride_n = d_out * h_out * w_out * 3;
    let g_stride_d = h_out * w_out * 3;
    let g_stride_h = w_out * 3;

    let o_stride_n = channels * d_out * h_out * w_out;
    let o_stride_c = d_out * h_out * w_out;
    let o_stride_d = h_out * w_out;
    let o_stride_h = w_out;

    // See `grid_sample_2d_impl` for why every dtype is widened to f64 here.
    let to_f64 = |x: T| -> f64 { ToElement::to_f64(&x) };
    let from_f64 = |x: f64| -> T {
        <T as NumCast>::from(x).unwrap_or_else(|| {
            panic!(
                "grid_sample_3d: NumCast::from({x:?}) to {:?} returned None",
                T::dtype()
            )
        })
    };

    // Read one voxel, resolving an out-of-range index according to the padding mode.
    let read = |t_base: usize, [xi, yi, zi]: [i64; 3]| -> f64 {
        match pad_mode {
            GridSamplePaddingMode::Zeros => {
                if xi >= 0
                    && xi < w_in as i64
                    && yi >= 0
                    && yi < h_in as i64
                    && zi >= 0
                    && zi < d_in as i64
                {
                    to_f64(
                        tensor_data[t_base
                            + zi as usize * t_stride_d
                            + yi as usize * t_stride_h
                            + xi as usize],
                    )
                } else {
                    0.0
                }
            }
            GridSamplePaddingMode::Border | GridSamplePaddingMode::Reflection => {
                let xi = xi.clamp(0, (w_in - 1) as i64) as usize;
                let yi = yi.clamp(0, (h_in - 1) as i64) as usize;
                let zi = zi.clamp(0, (d_in - 1) as i64) as usize;
                to_f64(tensor_data[t_base + zi * t_stride_d + yi * t_stride_h + xi])
            }
        }
    };

    for b in 0..batch_size {
        for z in 0..d_out {
            for y in 0..h_out {
                for x in 0..w_out {
                    let g_idx = b * g_stride_n + z * g_stride_d + y * g_stride_h + x * 3;
                    let sample_x = to_f64(grid_data[g_idx]);
                    let sample_y = to_f64(grid_data[g_idx + 1]);
                    let sample_z = to_f64(grid_data[g_idx + 2]);

                    // Convert normalized grid coordinates [-1, 1] to voxel coordinates. Each axis
                    // is scaled by its own input extent: x by W_in, y by H_in and z by D_in.
                    let coords = if align {
                        [
                            (sample_x + 1.0) * ((w_in - 1) as f64) / 2.0,
                            (sample_y + 1.0) * ((h_in - 1) as f64) / 2.0,
                            (sample_z + 1.0) * ((d_in - 1) as f64) / 2.0,
                        ]
                    } else {
                        [
                            (sample_x + 1.0) * (w_in as f64) / 2.0 - 0.5,
                            (sample_y + 1.0) * (h_in as f64) / 2.0 - 0.5,
                            (sample_z + 1.0) * (d_in as f64) / 2.0 - 0.5,
                        ]
                    };

                    let coords = apply_padding_3d(coords, size, pad_mode, align);

                    for c in 0..channels {
                        let t_base = b * t_stride_n + c * t_stride_c;
                        let o_idx =
                            b * o_stride_n + c * o_stride_c + z * o_stride_d + y * o_stride_h + x;

                        let val = match coords {
                            // inf/nan coordinates sample as zero outside of border padding
                            None => 0.0,
                            Some([px, py, pz])
                                if matches!(options.mode, InterpolateMode::Nearest) =>
                            {
                                // Ties round to even, matching `float_round` and PyTorch.
                                read(
                                    t_base,
                                    [
                                        round_ties_even(px) as i64,
                                        round_ties_even(py) as i64,
                                        round_ties_even(pz) as i64,
                                    ],
                                )
                            }
                            Some([px, py, pz]) => {
                                // Trilinear
                                let x0 = px.floor();
                                let y0 = py.floor();
                                let z0 = pz.floor();

                                let x_frac = px - x0;
                                let y_frac = py - y0;
                                let z_frac = pz - z0;

                                let x_idx = [x0 as i64, (x0 as i64).saturating_add(1)];
                                let y_idx = [y0 as i64, (y0 as i64).saturating_add(1)];
                                let z_idx = [z0 as i64, (z0 as i64).saturating_add(1)];

                                // Per-axis weights: index 0 pairs with the lower corner, index 1
                                // with the upper one.
                                let x_weight = [1.0 - x_frac, x_frac];
                                let y_weight = [1.0 - y_frac, y_frac];
                                let z_weight = [1.0 - z_frac, z_frac];

                                let mut acc = 0.0;
                                for z_corner in 0..2 {
                                    for y_corner in 0..2 {
                                        for x_corner in 0..2 {
                                            let value = read(
                                                t_base,
                                                [x_idx[x_corner], y_idx[y_corner], z_idx[z_corner]],
                                            );
                                            acc += value
                                                * x_weight[x_corner]
                                                * y_weight[y_corner]
                                                * z_weight[z_corner];
                                        }
                                    }
                                }
                                acc
                            }
                        };

                        output[o_idx] = from_f64(val);
                    }
                }
            }
        }
    }

    let bytes = Bytes::from_elems(output);
    FlexTensor::new(bytes, Layout::contiguous(out_shape), T::dtype())
}

/// Apply the padding mode to a 3-D sampling position.
///
/// Returns `None` when the sample is defined to be zero regardless of the position, which is how
/// inf/nan coordinates are handled outside of border padding.
fn apply_padding_3d(
    [x, y, z]: [f64; 3],
    [width, height, depth]: [usize; 3],
    mode: GridSamplePaddingMode,
    align_corners: bool,
) -> Option<[f64; 3]> {
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return match mode {
            // Clamp to the center of the volume for inf/nan
            GridSamplePaddingMode::Border => Some([
                ((width - 1) / 2) as f64,
                ((height - 1) / 2) as f64,
                ((depth - 1) / 2) as f64,
            ]),
            GridSamplePaddingMode::Zeros | GridSamplePaddingMode::Reflection => None,
        };
    }

    Some(match mode {
        GridSamplePaddingMode::Zeros => [x, y, z],
        GridSamplePaddingMode::Border => [
            x.clamp(0.0, (width - 1) as f64),
            y.clamp(0.0, (height - 1) as f64),
            z.clamp(0.0, (depth - 1) as f64),
        ],
        GridSamplePaddingMode::Reflection => [
            reflect_coordinate(x, width, align_corners),
            reflect_coordinate(y, height, align_corners),
            reflect_coordinate(z, depth, align_corners),
        ],
    })
}

/// Round half to even, matching `float_round` and PyTorch's `std::nearbyint`.
///
/// `f64::round_ties_even` is `std`-only and is not part of `num_traits::Float`, hence the
/// explicit correction.
fn round_ties_even(coord: f64) -> f64 {
    let rounded = coord.round();
    if (coord - rounded).abs() == 0.5 && rounded % 2.0 != 0.0 {
        rounded - rounded.signum()
    } else {
        rounded
    }
}

fn apply_padding(
    px: f64,
    py: f64,
    w: usize,
    h: usize,
    mode: GridSamplePaddingMode,
    align_corners: bool,
) -> (f64, f64) {
    if !px.is_finite() || !py.is_finite() {
        return match mode {
            GridSamplePaddingMode::Border => {
                let cx = ((w - 1) as f64 / 2.0).clamp(0.0, (w - 1) as f64);
                let cy = ((h - 1) as f64 / 2.0).clamp(0.0, (h - 1) as f64);
                (cx, cy)
            }
            _ => (px, py),
        };
    }

    match mode {
        GridSamplePaddingMode::Zeros => (px, py),
        GridSamplePaddingMode::Border => {
            let px = px.clamp(0.0, (w - 1) as f64);
            let py = py.clamp(0.0, (h - 1) as f64);
            (px, py)
        }
        GridSamplePaddingMode::Reflection => {
            let px = reflect_coordinate(px, w, align_corners);
            let py = reflect_coordinate(py, h, align_corners);
            (px, py)
        }
    }
}

fn reflect_coordinate(coord: f64, size: usize, align_corners: bool) -> f64 {
    let size_f = size as f64;
    let (min_val, max_val) = if align_corners {
        (0.0, size_f - 1.0)
    } else {
        (-0.5, size_f - 0.5)
    };

    let span = max_val - min_val;
    if span <= 0.0 {
        return min_val;
    }

    let period = 2.0 * span;
    let x = (coord - min_val).abs();
    let x_mod = x - (x / period).floor() * period;
    span - (x_mod - span).abs() + min_val
}

// Correctness of grid_sample_2d and grid_sample_3d (nearest/linear,
// zeros/border/reflection padding, align_corners on/off) is covered by the
// cross-backend suite in crates/burn-backend-tests/tests/tensor/float/ops/
// (grid_sample.rs and grid_sample_3d.rs), which reaches both via the
// `FloatTensorOps` trait for flex. No flex-specific tests remain here.
