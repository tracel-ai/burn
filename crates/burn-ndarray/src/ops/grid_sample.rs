use burn_backend::ElementConversion;
use burn_backend::ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use ndarray::{Array4, Array5};

use crate::SharedArray;
use crate::{FloatNdArrayElement, UnsafeSharedRef, iter_range_par, run_par};

/// Sample a tensor using grid-based sampling.
///
/// # Arguments
///
/// * `tensor` - The tensor being sampled from, must be contiguous with shape (N, C, H_in, W_in)
/// * `grid` - A tensor of locations, with shape (N, H_out, W_out, 2). Values are [-1, 1].
///   A [x = -1, y = -1] means top-left, and [x = 1, y = 1] means bottom-right
/// * `options` - Grid sampling options (mode, padding_mode, align_corners)
///
/// # Returns
///
/// A tensor with shape (N, C, H_out, W_out)
pub(crate) fn grid_sample_2d<E: FloatNdArrayElement>(
    tensor: SharedArray<E>,
    grid: SharedArray<E>,
    options: GridSampleOptions,
) -> SharedArray<E> {
    match options.mode {
        InterpolateMode::Bilinear => (),
        _ => todo!(
            "grid_sample_2d with {:?} mode is not implemented",
            options.mode
        ),
    }

    let tensor = tensor.into_dimensionality::<ndarray::Ix4>().unwrap();
    let grid = grid.into_dimensionality::<ndarray::Ix4>().unwrap();

    let (batch_size, channels, height_in, width_in) = tensor.dim();
    let (b, height_out, width_out, d) = grid.dim();
    assert!(batch_size == b);
    assert!(2 == d);

    let mut output = Array4::zeros((batch_size, channels, height_out, width_out));
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    let sample_count = batch_size * channels * height_out * width_out;
    let strides = (
        channels * height_out * width_out,
        height_out * width_out,
        width_out,
    );

    let align = options.align_corners;
    let pad_mode = options.padding_mode;

    run_par!(|| {
        iter_range_par!(0, sample_count).for_each(|id| {
            let (b, c, y, x) = (
                id / strides.0,
                id % strides.0 / strides.1,
                id % strides.1 / strides.2,
                id % strides.2,
            );

            let sample_x = grid[(b, y, x, 0)].elem::<f64>();
            let sample_y = grid[(b, y, x, 1)].elem::<f64>();

            // Convert normalized grid coordinates [-1, 1] to pixel coordinates
            let (px, py) = if align {
                // align_corners=true: x_pixel = (x_norm + 1) * (width - 1) / 2
                // Maps -1 to 0 and 1 to width - 1
                let px = (sample_x + 1.0) * ((width_in - 1) as f64) / 2.0;
                let py = (sample_y + 1.0) * ((height_in - 1) as f64) / 2.0;
                (px, py)
            } else {
                // align_corners=false: x_pixel = (x_norm + 1) * width / 2 - 0.5
                // Maps -1 to -0.5 and 1 to width - 0.5
                let px = (sample_x + 1.0) * (width_in as f64) / 2.0 - 0.5;
                let py = (sample_y + 1.0) * (height_in as f64) / 2.0 - 0.5;
                (px, py)
            };

            // Bilinear interpolation with the specified padding mode
            let val =
                bilinear_interpolate(&tensor, b, c, px, py, width_in, height_in, pad_mode, align);

            unsafe {
                let mut output = unsafe_shared_out.get();
                output[(b, c, y, x)] = val.elem();
            }
        });
    });

    output.into_dyn().into_shared()
}

/// Bilinear interpolation at a point with configurable padding mode.
#[allow(clippy::too_many_arguments)]
fn bilinear_interpolate<E, S>(
    source: &ndarray::ArrayBase<S, ndarray::Dim<[usize; 4]>>,
    b: usize,
    c: usize,
    x: f64,
    y: f64,
    width: usize,
    height: usize,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> f64
where
    E: FloatNdArrayElement,
    S: ndarray::Data<Elem = E>,
{
    // Handle inf/nan coordinates
    if !x.is_finite() || !y.is_finite() {
        return match padding_mode {
            GridSamplePaddingMode::Zeros => 0.0,
            GridSamplePaddingMode::Border => {
                // Clamp to center of image for inf/nan
                let cx = ((width - 1) as f64 / 2.0).clamp(0.0, (width - 1) as f64);
                let cy = ((height - 1) as f64 / 2.0).clamp(0.0, (height - 1) as f64);
                source[(b, c, cy as usize, cx as usize)].elem::<f64>()
            }
            GridSamplePaddingMode::Reflection => 0.0, // Simplified: treat as zeros for inf/nan
        };
    }

    // Apply padding mode to get actual sampling coordinates
    let (x, y) = match padding_mode {
        GridSamplePaddingMode::Border => {
            // Clamp coordinates to valid range [0, size-1]
            let x = x.clamp(0.0, (width - 1) as f64);
            let y = y.clamp(0.0, (height - 1) as f64);
            (x, y)
        }
        GridSamplePaddingMode::Reflection => {
            // Reflect coordinates at boundaries
            let x = reflect_coordinate(x, width, align_corners);
            let y = reflect_coordinate(y, height, align_corners);
            (x, y)
        }
        GridSamplePaddingMode::Zeros => (x, y), // Keep as-is, handle out-of-bounds in read
    };

    // Get the four corner indices
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0.saturating_add(1);
    let y1 = y0.saturating_add(1);

    // Compute interpolation weights (fractional part)
    let x_frac = x - x.floor();
    let y_frac = y - y.floor();

    // Helper to read a value based on padding mode
    let read_value = |xi: i64, yi: i64| -> f64 {
        match padding_mode {
            GridSamplePaddingMode::Zeros => {
                // Return 0 for out-of-bounds
                if xi >= 0 && xi < width as i64 && yi >= 0 && yi < height as i64 {
                    source[(b, c, yi as usize, xi as usize)].elem::<f64>()
                } else {
                    0.0
                }
            }
            GridSamplePaddingMode::Border | GridSamplePaddingMode::Reflection => {
                // Coordinates should already be in valid range after clamping/reflection
                let xi = xi.clamp(0, (width - 1) as i64) as usize;
                let yi = yi.clamp(0, (height - 1) as i64) as usize;
                source[(b, c, yi, xi)].elem::<f64>()
            }
        }
    };

    // Read the four corners
    let v00 = read_value(x0, y0);
    let v01 = read_value(x0, y1);
    let v10 = read_value(x1, y0);
    let v11 = read_value(x1, y1);

    // Bilinear interpolation weights
    let w00 = (1.0 - x_frac) * (1.0 - y_frac);
    let w01 = (1.0 - x_frac) * y_frac;
    let w10 = x_frac * (1.0 - y_frac);
    let w11 = x_frac * y_frac;

    v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
}

/// Reflect a coordinate at the boundaries using a triangle wave pattern.
///
/// For align_corners=true: reflects within [0, size-1]
/// For align_corners=false: reflects within [-0.5, size-0.5]
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

    // Triangle wave formula: span - |((x mod 2*span) - span)|
    let period = 2.0 * span;
    let x = (coord - min_val).abs();
    let x_mod = x - (x / period).floor() * period;
    span - (x_mod - span).abs() + min_val
}

/// Sample a tensor using 3-D grid-based sampling.
///
/// # Arguments
///
/// * `tensor` - The tensor being sampled from, must be contiguous with shape
///   (N, C, D_in, H_in, W_in)
/// * `grid` - A tensor of locations, with shape (N, D_out, H_out, W_out, 3). Values are [-1, 1]
///   and the last dimension is ordered `(x, y, z)`, where `x` indexes `W_in`, `y` indexes `H_in`
///   and `z` indexes `D_in`
/// * `options` - Grid sampling options (mode, padding_mode, align_corners)
///
/// # Returns
///
/// A tensor with shape (N, C, D_out, H_out, W_out)
pub(crate) fn grid_sample_3d<E: FloatNdArrayElement>(
    tensor: SharedArray<E>,
    grid: SharedArray<E>,
    options: GridSampleOptions,
) -> SharedArray<E> {
    let nearest = match options.mode {
        // `InterpolateMode` is shared across ranks, so `Bilinear` means trilinear here.
        InterpolateMode::Bilinear => false,
        InterpolateMode::Nearest => true,
        _ => todo!(
            "grid_sample_3d with {:?} mode is not implemented",
            options.mode
        ),
    };

    let tensor = tensor.into_dimensionality::<ndarray::Ix5>().unwrap();
    let grid = grid.into_dimensionality::<ndarray::Ix5>().unwrap();

    let (batch_size, channels, depth_in, height_in, width_in) = tensor.dim();
    let (b, depth_out, height_out, width_out, d) = grid.dim();
    assert!(batch_size == b);
    assert!(3 == d);

    let mut output = Array5::zeros((batch_size, channels, depth_out, height_out, width_out));
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    let sample_count = batch_size * channels * depth_out * height_out * width_out;
    let strides = (
        channels * depth_out * height_out * width_out,
        depth_out * height_out * width_out,
        height_out * width_out,
        width_out,
    );

    let align = options.align_corners;
    let pad_mode = options.padding_mode;
    let size = [width_in, height_in, depth_in];

    run_par!(|| {
        iter_range_par!(0, sample_count).for_each(|id| {
            let (b, c, z, y, x) = (
                id / strides.0,
                id % strides.0 / strides.1,
                id % strides.1 / strides.2,
                id % strides.2 / strides.3,
                id % strides.3,
            );

            let sample_x = grid[(b, z, y, x, 0)].elem::<f64>();
            let sample_y = grid[(b, z, y, x, 1)].elem::<f64>();
            let sample_z = grid[(b, z, y, x, 2)].elem::<f64>();

            // Convert normalized grid coordinates [-1, 1] to voxel coordinates. Each axis is
            // scaled by its own input extent: x by W_in, y by H_in and z by D_in.
            let coords = if align {
                // align_corners=true: x_voxel = (x_norm + 1) * (width - 1) / 2
                // Maps -1 to 0 and 1 to width - 1
                [
                    (sample_x + 1.0) * ((width_in - 1) as f64) / 2.0,
                    (sample_y + 1.0) * ((height_in - 1) as f64) / 2.0,
                    (sample_z + 1.0) * ((depth_in - 1) as f64) / 2.0,
                ]
            } else {
                // align_corners=false: x_voxel = (x_norm + 1) * width / 2 - 0.5
                // Maps -1 to -0.5 and 1 to width - 0.5
                [
                    (sample_x + 1.0) * (width_in as f64) / 2.0 - 0.5,
                    (sample_y + 1.0) * (height_in as f64) / 2.0 - 0.5,
                    (sample_z + 1.0) * (depth_in as f64) / 2.0 - 0.5,
                ]
            };

            let val = match apply_padding_3d(coords, size, pad_mode, align) {
                Some(coords) if nearest => nearest_sample_3d(&tensor, b, c, coords, size, pad_mode),
                Some(coords) => trilinear_interpolate(&tensor, b, c, coords, size, pad_mode),
                None => 0.0,
            };

            unsafe {
                let mut output = unsafe_shared_out.get();
                output[(b, c, z, y, x)] = val.elem();
            }
        });
    });

    output.into_dyn().into_shared()
}

/// Trilinear interpolation at a point whose coordinates already went through [`apply_padding_3d`].
fn trilinear_interpolate<E, S>(
    source: &ndarray::ArrayBase<S, ndarray::Dim<[usize; 5]>>,
    b: usize,
    c: usize,
    [x, y, z]: [f64; 3],
    size: [usize; 3],
    padding_mode: GridSamplePaddingMode,
) -> f64
where
    E: FloatNdArrayElement,
    S: ndarray::Data<Elem = E>,
{
    // Get the lower corner indices
    let x0 = x.floor();
    let y0 = y.floor();
    let z0 = z.floor();

    // Compute interpolation weights (fractional part)
    let x_frac = x - x0;
    let y_frac = y - y0;
    let z_frac = z - z0;

    let x_idx = [x0 as i64, (x0 as i64).saturating_add(1)];
    let y_idx = [y0 as i64, (y0 as i64).saturating_add(1)];
    let z_idx = [z0 as i64, (z0 as i64).saturating_add(1)];

    // Per-axis weights: index 0 pairs with the lower corner, index 1 with the upper one.
    let x_weight = [1.0 - x_frac, x_frac];
    let y_weight = [1.0 - y_frac, y_frac];
    let z_weight = [1.0 - z_frac, z_frac];

    let mut acc = 0.0;
    for z_corner in 0..2 {
        for y_corner in 0..2 {
            for x_corner in 0..2 {
                let value = read_voxel(
                    source,
                    b,
                    c,
                    [x_idx[x_corner], y_idx[y_corner], z_idx[z_corner]],
                    size,
                    padding_mode,
                );
                acc += value * x_weight[x_corner] * y_weight[y_corner] * z_weight[z_corner];
            }
        }
    }

    acc
}

/// Nearest-neighbor sampling at a point whose coordinates already went through
/// [`apply_padding_3d`].
fn nearest_sample_3d<E, S>(
    source: &ndarray::ArrayBase<S, ndarray::Dim<[usize; 5]>>,
    b: usize,
    c: usize,
    [x, y, z]: [f64; 3],
    size: [usize; 3],
    padding_mode: GridSamplePaddingMode,
) -> f64
where
    E: FloatNdArrayElement,
    S: ndarray::Data<Elem = E>,
{
    let idx = [
        round_ties_even(x) as i64,
        round_ties_even(y) as i64,
        round_ties_even(z) as i64,
    ];
    read_voxel(source, b, c, idx, size, padding_mode)
}

/// Apply the padding mode to a 3-D sampling position.
///
/// Returns `None` when the sample is defined to be zero regardless of the position, which is how
/// inf/nan coordinates are handled outside of border padding.
fn apply_padding_3d(
    [x, y, z]: [f64; 3],
    [width, height, depth]: [usize; 3],
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> Option<[f64; 3]> {
    // Handle inf/nan coordinates
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return match padding_mode {
            // Clamp to the center of the volume for inf/nan
            GridSamplePaddingMode::Border => Some([
                ((width - 1) / 2) as f64,
                ((height - 1) / 2) as f64,
                ((depth - 1) / 2) as f64,
            ]),
            // Simplified: treat as zeros for inf/nan
            GridSamplePaddingMode::Zeros | GridSamplePaddingMode::Reflection => None,
        };
    }

    Some(match padding_mode {
        // Clamp coordinates to valid range [0, size-1]
        GridSamplePaddingMode::Border => [
            x.clamp(0.0, (width - 1) as f64),
            y.clamp(0.0, (height - 1) as f64),
            z.clamp(0.0, (depth - 1) as f64),
        ],
        // Reflect coordinates at boundaries
        GridSamplePaddingMode::Reflection => [
            reflect_coordinate(x, width, align_corners),
            reflect_coordinate(y, height, align_corners),
            reflect_coordinate(z, depth, align_corners),
        ],
        // Keep as-is, handle out-of-bounds in read
        GridSamplePaddingMode::Zeros => [x, y, z],
    })
}

/// Read a voxel based on padding mode.
fn read_voxel<E, S>(
    source: &ndarray::ArrayBase<S, ndarray::Dim<[usize; 5]>>,
    b: usize,
    c: usize,
    [xi, yi, zi]: [i64; 3],
    [width, height, depth]: [usize; 3],
    padding_mode: GridSamplePaddingMode,
) -> f64
where
    E: FloatNdArrayElement,
    S: ndarray::Data<Elem = E>,
{
    match padding_mode {
        GridSamplePaddingMode::Zeros => {
            // Return 0 for out-of-bounds
            if xi >= 0
                && xi < width as i64
                && yi >= 0
                && yi < height as i64
                && zi >= 0
                && zi < depth as i64
            {
                source[(b, c, zi as usize, yi as usize, xi as usize)].elem::<f64>()
            } else {
                0.0
            }
        }
        GridSamplePaddingMode::Border | GridSamplePaddingMode::Reflection => {
            // Coordinates should already be in valid range after clamping/reflection
            let xi = xi.clamp(0, (width - 1) as i64) as usize;
            let yi = yi.clamp(0, (height - 1) as i64) as usize;
            let zi = zi.clamp(0, (depth - 1) as i64) as usize;
            source[(b, c, zi, yi, xi)].elem::<f64>()
        }
    }
}

/// Round half to even, matching `float_round` and PyTorch's `std::nearbyint`.
///
/// `f64::round_ties_even` is `std`-only and is not part of `num_traits::Float`, which is what
/// this crate falls back to in `no_std` builds — hence the explicit correction.
fn round_ties_even(coord: f64) -> f64 {
    let rounded = coord.round();
    if (coord - rounded).abs() == 0.5 && rounded % 2.0 != 0.0 {
        rounded - rounded.signum()
    } else {
        rounded
    }
}
