use cubecl::prelude::*;

use crate::{CubeRuntime, tensor::CubeTensor};
use burn_backend::ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode};

use super::bilinear::grid_sample_bilinear_launch;

/// Grid sample operation supporting bilinear interpolation
pub fn grid_sample<R: CubeRuntime>(
    input: CubeTensor<R>,
    grid: CubeTensor<R>,
    options: GridSampleOptions,
) -> CubeTensor<R> {
    match options.mode {
        InterpolateMode::Bilinear => grid_sample_bilinear_launch(input, grid, options),
        _ => panic!(
            "Unsupported grid_sample interpolation mode: {:?}",
            options.mode
        ),
    }
}

/// Compile-time padding mode for kernel specialization
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PaddingMode {
    /// Fill with zeros for out-of-bounds coordinates.
    Zeros,
    /// Clamp coordinates to the border (use nearest edge value).
    Border,
    /// Reflect coordinates at the boundary.
    Reflection,
}

impl From<GridSamplePaddingMode> for PaddingMode {
    fn from(mode: GridSamplePaddingMode) -> Self {
        match mode {
            GridSamplePaddingMode::Zeros => PaddingMode::Zeros,
            GridSamplePaddingMode::Border => PaddingMode::Border,
            GridSamplePaddingMode::Reflection => PaddingMode::Reflection,
        }
    }
}

/// Fetch value based on padding mode (dispatch to appropriate handler)
#[cube]
pub(crate) fn fetch_value<F: Float>(
    input: &Tensor<F>,
    base: u32,
    stride_h: u32,
    stride_w: u32,
    y: i32,
    x: i32,
    h: i32,
    w: i32,
    #[comptime] padding_mode: PaddingMode,
) -> F {
    match padding_mode {
        PaddingMode::Zeros => fetch_with_zeros(input, base, stride_h, stride_w, y, x, h, w),
        PaddingMode::Border => fetch_with_border(input, base, stride_h, stride_w, y, x, h, w),
        PaddingMode::Reflection => {
            fetch_with_reflection(input, base, stride_h, stride_w, y, x, h, w)
        }
    }
}

/// Fetch value with zeros padding (return 0 for out-of-bounds).
#[cube]
pub(crate) fn fetch_with_zeros<F: Float>(
    input: &Tensor<F>,
    base: u32,
    stride_h: u32,
    stride_w: u32,
    y: i32,
    x: i32,
    h: i32,
    w: i32,
) -> F {
    let in_bounds = x >= 0 && x < w && y >= 0 && y < h;
    let idx = base + (y as u32) * stride_h + (x as u32) * stride_w;
    select(in_bounds, input[idx], F::new(0.0))
}

/// Fetch value with border padding (clamp to edge).
#[cube]
pub(crate) fn fetch_with_border<F: Float>(
    input: &Tensor<F>,
    base: u32,
    stride_h: u32,
    stride_w: u32,
    y: i32,
    x: i32,
    h: i32,
    w: i32,
) -> F {
    let x_clamped = Min::min(Max::max(x, 0), w - 1) as u32;
    let y_clamped = Min::min(Max::max(y, 0), h - 1) as u32;
    let idx = base + y_clamped * stride_h + x_clamped * stride_w;
    input[idx]
}

/// Fetch value with reflection padding.
/// Assumes float reflection was applied to center, so indices are at most 2 steps out of bounds.
#[cube]
pub(crate) fn fetch_with_reflection<F: Float>(
    input: &Tensor<F>,
    base: u32,
    stride_h: u32,
    stride_w: u32,
    y: i32,
    x: i32,
    h: i32,
    w: i32,
) -> F {
    let x_reflected = reflect_coord_bounded(x, w);
    let y_reflected = reflect_coord_bounded(y, h);
    let idx = base + y_reflected * stride_h + x_reflected * stride_w;
    input[idx]
}

/// Reflect an integer index that may be out of bounds.
/// After float reflection, indices can be up to 2 steps out for bicubic (1 step for bilinear).
#[cube]
fn reflect_coord_bounded(idx: i32, size: i32) -> u32 {
    let max_idx = size - 1;
    let neg_reflected = -idx - 1;
    let pos_reflected = 2 * max_idx + 1 - idx;
    let result = select(
        idx < 0,
        neg_reflected,
        select(idx > max_idx, pos_reflected, idx),
    );
    Min::min(Max::max(result, 0), max_idx) as u32
}

/// Reflect a float coordinate into the valid sampling range.
#[cube]
pub(crate) fn reflect_coord<F: Float>(coord: F, size: u32, #[comptime] align_corners: bool) -> F {
    let size_f = F::cast_from(size);
    if align_corners {
        reflect_float_impl::<F>(coord, F::new(0.0), size_f - F::new(1.0))
    } else {
        reflect_float_impl::<F>(coord, F::new(-0.5), size_f - F::new(0.5))
    }
}

/// Reflect a float coordinate into [min_val, max_val] using a triangle wave pattern.
#[cube]
fn reflect_float_impl<F: Float>(coord: F, min_val: F, max_val: F) -> F {
    let span = max_val - min_val;

    let is_valid = span > F::new(0.0);
    let safe_span = select(is_valid, span, F::new(1.0));

    // Triangle wave formula: span - |((x mod 2*span) - span)| + min_val
    let period = safe_span * F::new(2.0);
    let x = Abs::abs(coord - min_val);
    let x_mod = x - Floor::floor(x / period) * period;
    let reflected = safe_span - Abs::abs(x_mod - safe_span) + min_val;

    select(is_valid, reflected, min_val)
}
