//! Interpolation operations for image resizing.
//!
//! Supported modes:
//! - Nearest: Floor-based coordinate mapping (fastest)
//! - Bilinear: 4-point weighted average (good quality/speed tradeoff)
//! - Bicubic: 16-point cubic convolution (highest quality)
//!
//! Optimizations:
//! - Rayon parallelism over (batch, channel) pairs
//! - Precomputed coordinate mappings where beneficial
//!
//! Supported dtypes: f32, f64, f16 (native), bf16 (via f32 conversion)

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_backend::ops::{
    CoordinateTransformMode, CubicOptions, InterpolateOptions, NearestRoundingMode,
};
use burn_std::{Bytes, Shape, bf16, f16};
use num_traits::Float;

use crate::{FlexTensor, Layout};

// ============================================================================
// Macros for dtype wrappers
// ============================================================================

/// Generates an interpolation forward typed dispatcher.
macro_rules! interpolate_typed {
    ($fn_name:ident, $impl_fn:ident, $T:ty) => {
        pub fn $fn_name(
            x: FlexTensor,
            output_size: [usize; 2],
            options: &InterpolateOptions,
        ) -> FlexTensor {
            $impl_fn::<$T>(x, output_size, options)
        }
    };
}

/// Generates an interpolation bf16 forward wrapper via f32 conversion.
macro_rules! interpolate_bf16 {
    ($bf16_fn:ident, $f32_fn:ident) => {
        pub fn $bf16_fn(
            x: FlexTensor,
            output_size: [usize; 2],
            options: &InterpolateOptions,
        ) -> FlexTensor {
            let x_f32 = convert_bf16_to_f32(&x);
            let result_f32 = $f32_fn(x_f32, output_size, options);
            convert_f32_to_bf16(&result_f32)
        }
    };
}

/// Generates an interpolation backward typed dispatcher.
macro_rules! interpolate_backward_typed {
    ($fn_name:ident, $impl_fn:ident, $T:ty) => {
        pub fn $fn_name(
            x: FlexTensor,
            grad: FlexTensor,
            output_size: [usize; 2],
            options: &InterpolateOptions,
        ) -> FlexTensor {
            $impl_fn::<$T>(x, grad, output_size, options)
        }
    };
}

/// Generates an interpolation bf16 backward wrapper via f32 conversion.
macro_rules! interpolate_backward_bf16 {
    ($bf16_fn:ident, $f32_fn:ident) => {
        pub fn $bf16_fn(
            x: FlexTensor,
            grad: FlexTensor,
            output_size: [usize; 2],
            options: &InterpolateOptions,
        ) -> FlexTensor {
            let x_f32 = convert_bf16_to_f32(&x);
            let grad_f32 = convert_bf16_to_f32(&grad);
            let result_f32 = $f32_fn(x_f32, grad_f32, output_size, options);
            convert_f32_to_bf16(&result_f32)
        }
    };
}

// ============================================================================
// Public API - dtype dispatch
// ============================================================================

interpolate_typed!(interpolate_nearest_f32, interpolate_nearest_impl, f32);
interpolate_typed!(interpolate_nearest_f64, interpolate_nearest_impl, f64);
interpolate_typed!(interpolate_nearest_f16, interpolate_nearest_impl, f16);
interpolate_bf16!(interpolate_nearest_bf16, interpolate_nearest_f32);

interpolate_typed!(interpolate_bilinear_f32, interpolate_bilinear_impl, f32);
interpolate_typed!(interpolate_bilinear_f64, interpolate_bilinear_impl, f64);
interpolate_typed!(interpolate_bilinear_f16, interpolate_bilinear_impl, f16);
interpolate_bf16!(interpolate_bilinear_bf16, interpolate_bilinear_f32);

interpolate_typed!(interpolate_bicubic_f32, interpolate_bicubic_impl, f32);
interpolate_typed!(interpolate_bicubic_f64, interpolate_bicubic_impl, f64);
interpolate_typed!(interpolate_bicubic_f16, interpolate_bicubic_impl, f16);
interpolate_bf16!(interpolate_bicubic_bf16, interpolate_bicubic_f32);

interpolate_typed!(interpolate_lanczos3_f32, interpolate_lanczos3_impl, f32);
interpolate_typed!(interpolate_lanczos3_f64, interpolate_lanczos3_impl, f64);
interpolate_typed!(interpolate_lanczos3_f16, interpolate_lanczos3_impl, f16);
interpolate_bf16!(interpolate_lanczos3_bf16, interpolate_lanczos3_f32);

// ============================================================================
// Backward pass - dtype dispatch
// ============================================================================

interpolate_backward_typed!(
    interpolate_nearest_backward_f32,
    interpolate_nearest_backward_impl,
    f32
);
interpolate_backward_typed!(
    interpolate_nearest_backward_f64,
    interpolate_nearest_backward_impl,
    f64
);
interpolate_backward_typed!(
    interpolate_nearest_backward_f16,
    interpolate_nearest_backward_impl,
    f16
);
interpolate_backward_bf16!(
    interpolate_nearest_backward_bf16,
    interpolate_nearest_backward_f32
);

interpolate_backward_typed!(
    interpolate_bilinear_backward_f32,
    interpolate_bilinear_backward_impl,
    f32
);
interpolate_backward_typed!(
    interpolate_bilinear_backward_f64,
    interpolate_bilinear_backward_impl,
    f64
);
interpolate_backward_typed!(
    interpolate_bilinear_backward_f16,
    interpolate_bilinear_backward_impl,
    f16
);
interpolate_backward_bf16!(
    interpolate_bilinear_backward_bf16,
    interpolate_bilinear_backward_f32
);

interpolate_backward_typed!(
    interpolate_bicubic_backward_f32,
    interpolate_bicubic_backward_impl,
    f32
);
interpolate_backward_typed!(
    interpolate_bicubic_backward_f64,
    interpolate_bicubic_backward_impl,
    f64
);
interpolate_backward_typed!(
    interpolate_bicubic_backward_f16,
    interpolate_bicubic_backward_impl,
    f16
);
interpolate_backward_bf16!(
    interpolate_bicubic_backward_bf16,
    interpolate_bicubic_backward_f32
);

// ============================================================================
// Generic implementations with rayon parallelism
// ============================================================================

/// Mapped coordinate on an axis, which can either be inside valid bounds or extrapolated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MappedCoord {
    /// In-bounds fractional coordinate.
    InBounds(f64),
    /// Out-of-bounds coordinate for TfCropAndResize with its extrapolation fill value.
    Extrapolated(f64),
}

/// Map an output coordinate to input coordinate for a given spatial axis.
pub fn map_coordinate(
    out_coord: usize,
    in_size: usize,
    out_size: usize,
    scale_factor: Option<f32>,
    mode: CoordinateTransformMode,
    axis: usize,
) -> MappedCoord {
    let s: f64 = match scale_factor {
        Some(s) => s as f64,
        None => out_size as f64 / in_size as f64,
    };
    let l_in = in_size as f64;
    let l_out = out_size as f64;
    let x_out = out_coord as f64;

    match mode {
        CoordinateTransformMode::HalfPixel => {
            let x_in = match scale_factor {
                Some(_) => (x_out + 0.5) / s - 0.5,
                None => (x_out + 0.5) * l_in / l_out - 0.5,
            };
            MappedCoord::InBounds(x_in)
        }
        CoordinateTransformMode::AlignCorners => {
            let x_in = if out_size == 1 {
                0.0
            } else {
                x_out * (l_in - 1.0) / (l_out - 1.0)
            };
            MappedCoord::InBounds(x_in)
        }
        CoordinateTransformMode::Asymmetric => {
            let x_in = match scale_factor {
                Some(_) => x_out / s,
                None => x_out * l_in / l_out,
            };
            MappedCoord::InBounds(x_in)
        }
        CoordinateTransformMode::PytorchHalfPixel => {
            if l_out > 1.0 {
                let x_in = match scale_factor {
                    Some(_) => (x_out + 0.5) / s - 0.5,
                    None => (x_out + 0.5) * l_in / l_out - 0.5,
                };
                MappedCoord::InBounds(x_in)
            } else {
                MappedCoord::InBounds(0.0)
            }
        }
        CoordinateTransformMode::HalfPixelSymmetric => {
            let x_in = match scale_factor {
                Some(_) => (x_out + 0.5 + (l_in * s - l_out) / 2.0) / s - 0.5,
                // With explicit sizes the centering offset is zero. Avoid rounding a
                // scale ratio before mapping coordinates, which can move nearest ties.
                None => (x_out + 0.5) * l_in / l_out - 0.5,
            };
            MappedCoord::InBounds(x_in)
        }
        CoordinateTransformMode::TfCropAndResize {
            roi,
            extrapolation_value,
        } => {
            let [r_start, r_end] = roi[axis];
            assert!(
                !r_start.is_nan() && !r_end.is_nan(),
                "TfCropAndResize: ROI coordinates must not be NaN, got [{}, {}]",
                r_start,
                r_end
            );
            let r_start = r_start as f64;
            let r_end = r_end as f64;
            let x_in = if out_size == 1 {
                0.5 * (r_start + r_end) * (l_in - 1.0)
            } else {
                r_start * (l_in - 1.0) + x_out * (r_end - r_start) * (l_in - 1.0) / (l_out - 1.0)
            };
            if x_in < 0.0 || x_in > (l_in - 1.0) {
                MappedCoord::Extrapolated(extrapolation_value as f64)
            } else {
                MappedCoord::InBounds(x_in)
            }
        }
    }
}

#[derive(Clone, Copy)]
enum NearestEntry {
    Index(usize),
    Extrapolated(f64),
}

fn nearest_axis_map(
    in_size: usize,
    out_size: usize,
    scale_factor: Option<f32>,
    coord_mode: CoordinateTransformMode,
    nearest_mode: NearestRoundingMode,
    axis: usize,
) -> Vec<NearestEntry> {
    (0..out_size)
        .map(|out_coord| {
            match map_coordinate(out_coord, in_size, out_size, scale_factor, coord_mode, axis) {
                MappedCoord::Extrapolated(val) => NearestEntry::Extrapolated(val),
                MappedCoord::InBounds(x_in) => {
                    let idx = match nearest_mode {
                        NearestRoundingMode::RoundPreferFloor => (x_in - 0.5).ceil() as isize,
                        NearestRoundingMode::RoundPreferCeil => (x_in + 0.5).floor() as isize,
                        NearestRoundingMode::Floor => x_in.floor() as isize,
                        NearestRoundingMode::Ceil => x_in.ceil() as isize,
                    };
                    let clamped = idx.clamp(0, in_size as isize - 1) as usize;
                    NearestEntry::Index(clamped)
                }
            }
        })
        .collect()
}

/// Nearest neighbor interpolation.
fn interpolate_nearest_impl<T>(
    x: FlexTensor,
    output_size: [usize; 2],
    options: &InterpolateOptions,
) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
{
    let x = x.to_contiguous();
    let input = x.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let scale_y = options.scale_factor.map(|s| s[0]);
    let scale_x = options.scale_factor.map(|s| s[1]);
    let y_map = nearest_axis_map(
        in_height,
        out_height,
        scale_y,
        options.coordinate_transform,
        options.nearest_mode,
        0,
    );
    let x_map = nearest_axis_map(
        in_width,
        out_width,
        scale_x,
        options.coordinate_transform,
        options.nearest_mode,
        1,
    );

    let out_numel = batch * channels * out_height * out_width;
    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    #[inline]
    fn gather_plane<T: Float + Copy>(
        input: &[T],
        in_base: usize,
        output: &mut [T],
        in_width: usize,
        out_width: usize,
        y_map: &[NearestEntry],
        x_map: &[NearestEntry],
    ) {
        for (oh, y_entry) in y_map.iter().enumerate() {
            let out_row_start = oh * out_width;
            match y_entry {
                NearestEntry::Extrapolated(val) => {
                    let ext = T::from(*val).unwrap();
                    for ow in 0..out_width {
                        output[out_row_start + ow] = ext;
                    }
                }
                NearestEntry::Index(ih) => {
                    let in_row = in_base + ih * in_width;
                    for (ow, x_entry) in x_map.iter().enumerate() {
                        output[out_row_start + ow] = match x_entry {
                            NearestEntry::Extrapolated(val) => T::from(*val).unwrap(),
                            NearestEntry::Index(iw) => input[in_row + iw],
                        };
                    }
                }
            }
        }
    }

    let output = {
        let mut output: Vec<T> = Vec::with_capacity(out_numel);
        #[allow(clippy::uninit_vec)]
        unsafe {
            output.set_len(out_numel);
        }

        let bc = batch * channels;

        #[cfg(feature = "rayon")]
        if out_numel >= super::PARALLEL_THRESHOLD {
            use rayon::prelude::*;

            output
                .par_chunks_mut(out_hw)
                .enumerate()
                .for_each(|(bc_idx, out_plane)| {
                    let in_base = bc_idx * in_hw;
                    gather_plane(
                        input, in_base, out_plane, in_width, out_width, &y_map, &x_map,
                    );
                });
        } else {
            for bc_idx in 0..bc {
                let in_base = bc_idx * in_hw;
                let out_start = bc_idx * out_hw;
                gather_plane(
                    input,
                    in_base,
                    &mut output[out_start..out_start + out_hw],
                    in_width,
                    out_width,
                    &y_map,
                    &x_map,
                );
            }
        }
        #[cfg(not(feature = "rayon"))]
        for bc_idx in 0..bc {
            let in_base = bc_idx * in_hw;
            let out_start = bc_idx * out_hw;
            gather_plane(
                input,
                in_base,
                &mut output[out_start..out_start + out_hw],
                in_width,
                out_width,
                &y_map,
                &x_map,
            );
        }
        output
    };

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(Shape::from(vec![batch, channels, out_height, out_width])),
        x.dtype(),
    )
}

#[derive(Clone, Copy)]
enum BilinearTap {
    InBounds {
        low: usize,
        high: usize,
        weight: f64,
    },
    Extrapolated(f64),
}

fn bilinear_axis_taps(
    in_size: usize,
    out_size: usize,
    scale_factor: Option<f32>,
    coord_mode: CoordinateTransformMode,
    axis: usize,
) -> Vec<BilinearTap> {
    (0..out_size)
        .map(|out_coord| {
            match map_coordinate(out_coord, in_size, out_size, scale_factor, coord_mode, axis) {
                MappedCoord::Extrapolated(val) => BilinearTap::Extrapolated(val),
                MappedCoord::InBounds(x_in) => {
                    let low = (x_in.floor().max(0.0) as usize).min(in_size - 1);
                    let high = (low + 1).min(in_size - 1);
                    let weight = (x_in - low as f64).max(0.0);
                    BilinearTap::InBounds { low, high, weight }
                }
            }
        })
        .collect()
}

/// Bilinear interpolation.
fn interpolate_bilinear_impl<T>(
    x: FlexTensor,
    output_size: [usize; 2],
    options: &InterpolateOptions,
) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
{
    let x = x.to_contiguous();
    let input = x.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let scale_y = options.scale_factor.map(|s| s[0]);
    let scale_x = options.scale_factor.map(|s| s[1]);
    let y_axis_taps = bilinear_axis_taps(
        in_height,
        out_height,
        scale_y,
        options.coordinate_transform,
        0,
    );
    let x_axis_taps = bilinear_axis_taps(
        in_width,
        out_width,
        scale_x,
        options.coordinate_transform,
        1,
    );

    let out_numel = batch * channels * out_height * out_width;
    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    #[inline]
    fn bilinear_sample<T: Float + Copy>(
        input: &[T],
        in_base: usize,
        in_width: usize,
        y_tap: &BilinearTap,
        x_tap: &BilinearTap,
    ) -> T {
        match (y_tap, x_tap) {
            (BilinearTap::Extrapolated(val), _) | (_, BilinearTap::Extrapolated(val)) => {
                T::from(*val).unwrap()
            }
            (
                BilinearTap::InBounds {
                    low: y_low,
                    high: y_high,
                    weight: y_weight,
                },
                BilinearTap::InBounds {
                    low: x_low,
                    high: x_high,
                    weight: x_weight,
                },
            ) => {
                let p_a = input[in_base + y_low * in_width + x_low];
                let p_b = input[in_base + y_low * in_width + x_high];
                let p_c = input[in_base + y_high * in_width + x_low];
                let p_d = input[in_base + y_high * in_width + x_high];

                let one = T::one();
                let xw = T::from(*x_weight).unwrap();
                let yw = T::from(*y_weight).unwrap();
                p_a * (one - xw) * (one - yw)
                    + p_b * xw * (one - yw)
                    + p_c * (one - xw) * yw
                    + p_d * xw * yw
            }
        }
    }

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![T::zero(); out_numel];
            let out_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());

            (0..batch).into_par_iter().for_each(|b| {
                (0..channels).into_par_iter().for_each(|c| {
                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    for (oh, y_tap) in y_axis_taps.iter().enumerate() {
                        for (ow, x_tap) in x_axis_taps.iter().enumerate() {
                            let result = bilinear_sample(input, in_base, in_width, y_tap, x_tap);
                            let out_idx = out_base + oh * out_width + ow;
                            unsafe {
                                out_ptr.write(out_idx, result);
                            }
                        }
                    }
                });
            });
            output
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![T::zero(); out_numel];

            for b in 0..batch {
                for c in 0..channels {
                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    for (oh, y_tap) in y_axis_taps.iter().enumerate() {
                        for (ow, x_tap) in x_axis_taps.iter().enumerate() {
                            let result = bilinear_sample(input, in_base, in_width, y_tap, x_tap);
                            output[out_base + oh * out_width + ow] = result;
                        }
                    }
                }
            }
            output
        }
    };

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(Shape::from(vec![batch, channels, out_height, out_width])),
        x.dtype(),
    )
}

#[derive(Clone, Copy)]
struct AxisTap {
    index: usize,
    weight: f64,
}

#[derive(Clone, Copy)]
struct AxisTaps<const N: usize> {
    taps: [AxisTap; N],
    len: usize,
}

impl<const N: usize> AxisTaps<N> {
    #[inline]
    fn as_slice(&self) -> &[AxisTap] {
        &self.taps[..self.len]
    }
}

#[derive(Clone, Copy)]
enum BicubicTap {
    InBounds([AxisTap; 4]),
    Extrapolated(f64),
}

/// Precompute Bicubic source indices and weights for one output axis.
fn bicubic_axis_taps(
    in_size: usize,
    out_size: usize,
    scale_factor: Option<f32>,
    coord_mode: CoordinateTransformMode,
    cubic_options: CubicOptions,
    axis: usize,
) -> Vec<BicubicTap> {
    let a = cubic_options.a as f64;
    let exclude_outside = cubic_options.exclude_outside;

    (0..out_size)
        .map(|out_coord| {
            match map_coordinate(out_coord, in_size, out_size, scale_factor, coord_mode, axis) {
                MappedCoord::Extrapolated(val) => BicubicTap::Extrapolated(val),
                MappedCoord::InBounds(coord) => {
                    let base = coord.floor() as isize;
                    let mut taps = [AxisTap {
                        index: 0,
                        weight: 0.0,
                    }; 4];

                    if exclude_outside {
                        let mut sum = 0.0_f64;
                        for (tap, offset) in taps.iter_mut().zip(-1..=2_isize) {
                            let raw_idx = base + offset;
                            let distance = (coord - base as f64) - offset as f64;
                            if raw_idx >= 0 && raw_idx < in_size as isize {
                                let w = cubic_weight(distance, a);
                                *tap = AxisTap {
                                    index: raw_idx as usize,
                                    weight: w,
                                };
                                sum += w;
                            } else {
                                *tap = AxisTap {
                                    index: 0,
                                    weight: 0.0,
                                };
                            }
                        }

                        if sum != 0.0 && sum.is_finite() {
                            for tap in taps.iter_mut() {
                                tap.weight /= sum;
                            }
                        } else {
                            let nearest = if coord < 0.0 { 0 } else { in_size - 1 };
                            taps[0] = AxisTap {
                                index: nearest,
                                weight: 1.0,
                            };
                            taps[1].weight = 0.0;
                            taps[2].weight = 0.0;
                            taps[3].weight = 0.0;
                        }
                    } else {
                        for (tap, offset) in taps.iter_mut().zip(-1..=2_isize) {
                            let raw_idx = base + offset;
                            let index = raw_idx.clamp(0, in_size as isize - 1) as usize;
                            let distance = (coord - base as f64) - offset as f64;
                            *tap = AxisTap {
                                index,
                                weight: cubic_weight(distance, a),
                            };
                        }
                    }

                    BicubicTap::InBounds(taps)
                }
            }
        })
        .collect()
}

/// Bicubic interpolation using cubic convolution.
fn interpolate_bicubic_impl<T>(
    x: FlexTensor,
    output_size: [usize; 2],
    options: &InterpolateOptions,
) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
{
    let x = x.to_contiguous();
    let input = x.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let scale_y = options.scale_factor.map(|s| s[0]);
    let scale_x = options.scale_factor.map(|s| s[1]);
    let y_axis_taps = bicubic_axis_taps(
        in_height,
        out_height,
        scale_y,
        options.coordinate_transform,
        options.cubic_options,
        0,
    );
    let x_axis_taps = bicubic_axis_taps(
        in_width,
        out_width,
        scale_x,
        options.coordinate_transform,
        options.cubic_options,
        1,
    );

    let out_numel = batch * channels * out_height * out_width;
    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    #[inline]
    fn bicubic_sample<T: Float + burn_backend::Element + bytemuck::Pod>(
        input: &[T],
        in_base: usize,
        in_width: usize,
        y_tap: &BicubicTap,
        x_tap: &BicubicTap,
    ) -> T {
        match (y_tap, x_tap) {
            (BicubicTap::Extrapolated(val), _) | (_, BicubicTap::Extrapolated(val)) => {
                T::from(*val).unwrap()
            }
            (BicubicTap::InBounds(y_taps), BicubicTap::InBounds(x_taps)) => {
                let mut sum = 0.0_f64;

                for y_tap in y_taps {
                    if y_tap.weight == 0.0 {
                        continue;
                    }
                    for x_tap in x_taps {
                        if x_tap.weight == 0.0 {
                            continue;
                        }
                        let value = input[in_base + y_tap.index * in_width + x_tap.index];
                        let value = <T as num_traits::ToPrimitive>::to_f64(&value).unwrap_or(0.0);
                        sum += value * x_tap.weight * y_tap.weight;
                    }
                }

                T::from(sum).unwrap()
            }
        }
    }

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![T::zero(); out_numel];
            let out_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());
            let num_bc_pairs = batch * channels;

            // Adaptive parallelization: if few (batch, channel) pairs, parallelize rows too
            if num_bc_pairs < 8 {
                let total_rows = batch * channels * out_height;
                (0..total_rows).into_par_iter().for_each(|id| {
                    let b = id / (channels * out_height);
                    let remainder = id % (channels * out_height);
                    let c = remainder / out_height;
                    let oh = remainder % out_height;

                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    let y_tap = &y_axis_taps[oh];

                    for (ow, x_tap) in x_axis_taps.iter().enumerate() {
                        let out_idx = out_base + oh * out_width + ow;
                        unsafe {
                            out_ptr.write(
                                out_idx,
                                bicubic_sample(input, in_base, in_width, y_tap, x_tap),
                            );
                        }
                    }
                });
            } else {
                (0..batch).into_par_iter().for_each(|b| {
                    (0..channels).into_par_iter().for_each(|c| {
                        let in_base = b * channels * in_hw + c * in_hw;
                        let out_base = b * channels * out_hw + c * out_hw;

                        for (oh, y_tap) in y_axis_taps.iter().enumerate() {
                            for (ow, x_tap) in x_axis_taps.iter().enumerate() {
                                let out_idx = out_base + oh * out_width + ow;
                                unsafe {
                                    out_ptr.write(
                                        out_idx,
                                        bicubic_sample(input, in_base, in_width, y_tap, x_tap),
                                    );
                                }
                            }
                        }
                    });
                });
            }
            output
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![T::zero(); out_numel];

            for b in 0..batch {
                for c in 0..channels {
                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    for (oh, y_tap) in y_axis_taps.iter().enumerate() {
                        for (ow, x_tap) in x_axis_taps.iter().enumerate() {
                            output[out_base + oh * out_width + ow] =
                                bicubic_sample(input, in_base, in_width, y_tap, x_tap);
                        }
                    }
                }
            }
            output
        }
    };

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(Shape::from(vec![batch, channels, out_height, out_width])),
        x.dtype(),
    )
}

/// Cubic convolution weight function.
/// Uses the Keys cubic interpolation kernel.
#[inline]
fn cubic_weight(t: f64, a: f64) -> f64 {
    let t_abs = t.abs();
    if t_abs < 1.0 {
        ((a + 2.0) * t_abs - (a + 3.0)) * t_abs * t_abs + 1.0
    } else if t_abs < 2.0 {
        ((a * t_abs - 5.0 * a) * t_abs + 8.0 * a) * t_abs - 4.0 * a
    } else {
        0.0
    }
}

/// Lanczos3 interpolation weight function.
/// Uses a sinc-windowed sinc kernel with a=3.
#[inline]
fn lanczos3_weight(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let abs_x = x.abs();
    if abs_x >= 3.0 {
        return 0.0;
    }
    let pi = core::f64::consts::PI;
    let pi_x = pi * x;
    let pi_x_over_3 = pi_x / 3.0;
    (pi_x.sin() * pi_x_over_3.sin()) / (pi_x * pi_x_over_3)
}

#[derive(Clone, Copy)]
enum Lanczos3Tap {
    InBounds(AxisTaps<6>),
    Extrapolated(f64),
}

/// Precompute Lanczos3 source indices and weights for one output axis.
fn lanczos3_axis_taps(
    in_size: usize,
    out_size: usize,
    scale_factor: Option<f32>,
    coord_mode: CoordinateTransformMode,
    axis: usize,
) -> Vec<Lanczos3Tap> {
    (0..out_size)
        .map(|out_coord| {
            match map_coordinate(out_coord, in_size, out_size, scale_factor, coord_mode, axis) {
                MappedCoord::Extrapolated(val) => Lanczos3Tap::Extrapolated(val),
                MappedCoord::InBounds(coord) => {
                    let base = coord.floor();
                    let mut taps = [AxisTap {
                        index: 0,
                        weight: 0.0,
                    }; 6];
                    let mut len = 0;

                    for offset in -2..=3_isize {
                        let index = base as isize + offset;
                        if index < 0 || index >= in_size as isize {
                            continue;
                        }

                        taps[len] = AxisTap {
                            index: index as usize,
                            weight: lanczos3_weight(coord - (base + offset as f64)),
                        };
                        len += 1;
                    }

                    Lanczos3Tap::InBounds(AxisTaps { taps, len })
                }
            }
        })
        .collect()
}

/// Lanczos3 interpolation (6x6 sinc-windowed kernel).
///
/// Uses skip-and-renormalize boundary handling: out-of-bounds samples are
/// excluded and weights are renormalized. This avoids edge ringing artifacts
/// from replicated boundary pixels (unlike bicubic which clamps to edge).
fn interpolate_lanczos3_impl<T>(
    x: FlexTensor,
    output_size: [usize; 2],
    options: &InterpolateOptions,
) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
{
    let x = x.to_contiguous();
    let input = x.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let scale_y = options.scale_factor.map(|s| s[0]);
    let scale_x = options.scale_factor.map(|s| s[1]);
    let y_axis_taps = lanczos3_axis_taps(
        in_height,
        out_height,
        scale_y,
        options.coordinate_transform,
        0,
    );
    let x_axis_taps = lanczos3_axis_taps(
        in_width,
        out_width,
        scale_x,
        options.coordinate_transform,
        1,
    );

    let out_numel = batch * channels * out_height * out_width;
    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    #[inline]
    fn lanczos3_sample<T: Float + burn_backend::Element + bytemuck::Pod>(
        input: &[T],
        in_base: usize,
        in_width: usize,
        y_tap: &Lanczos3Tap,
        x_tap: &Lanczos3Tap,
    ) -> T {
        match (y_tap, x_tap) {
            (Lanczos3Tap::Extrapolated(val), _) | (_, Lanczos3Tap::Extrapolated(val)) => {
                T::from(*val).unwrap()
            }
            (Lanczos3Tap::InBounds(y_taps), Lanczos3Tap::InBounds(x_taps)) => {
                let mut result = 0.0_f64;
                let mut weight_sum = 0.0_f64;

                for y_tap in y_taps.as_slice() {
                    for x_tap in x_taps.as_slice() {
                        let weight = y_tap.weight * x_tap.weight;
                        let value = input[in_base + y_tap.index * in_width + x_tap.index];
                        let value = <T as num_traits::ToPrimitive>::to_f64(&value).unwrap_or(0.0);
                        result += value * weight;
                        weight_sum += weight;
                    }
                }

                if weight_sum != 0.0 {
                    result /= weight_sum;
                }
                T::from(result).unwrap()
            }
        }
    }

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![T::zero(); out_numel];
            let out_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());

            let total_rows = batch * channels * out_height;
            (0..total_rows).into_par_iter().for_each(|id| {
                let b = id / (channels * out_height);
                let remainder = id % (channels * out_height);
                let c = remainder / out_height;
                let oh = remainder % out_height;

                let in_base = b * channels * in_hw + c * in_hw;
                let out_base = b * channels * out_hw + c * out_hw;

                let y_tap = &y_axis_taps[oh];

                for (ow, x_tap) in x_axis_taps.iter().enumerate() {
                    let out_idx = out_base + oh * out_width + ow;
                    unsafe {
                        out_ptr.write(
                            out_idx,
                            lanczos3_sample(input, in_base, in_width, y_tap, x_tap),
                        );
                    }
                }
            });
            output
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![T::zero(); out_numel];

            for b in 0..batch {
                for c in 0..channels {
                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    for (oh, y_tap) in y_axis_taps.iter().enumerate() {
                        for (ow, x_tap) in x_axis_taps.iter().enumerate() {
                            output[out_base + oh * out_width + ow] =
                                lanczos3_sample(input, in_base, in_width, y_tap, x_tap);
                        }
                    }
                }
            }
            output
        }
    };

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(Shape::from(vec![batch, channels, out_height, out_width])),
        x.dtype(),
    )
}

// ============================================================================
// Backward implementations
// ============================================================================

/// Nearest neighbor backward: accumulates gradients at source positions.
fn interpolate_nearest_backward_impl<T>(
    x: FlexTensor,
    grad: FlexTensor,
    output_size: [usize; 2],
    options: &InterpolateOptions,
) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
{
    let grad = grad.to_contiguous();
    let grad_data = grad.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let scale_y = options.scale_factor.map(|s| s[0]);
    let scale_x = options.scale_factor.map(|s| s[1]);
    let y_map = nearest_axis_map(
        in_height,
        out_height,
        scale_y,
        options.coordinate_transform,
        options.nearest_mode,
        0,
    );
    let x_map = nearest_axis_map(
        in_width,
        out_width,
        scale_x,
        options.coordinate_transform,
        options.nearest_mode,
        1,
    );

    let in_numel = batch * channels * in_height * in_width;
    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    #[inline]
    fn scatter_plane<T: Float + Copy>(
        grad_data: &[T],
        grad_base: usize,
        input_grad: &mut [T],
        in_width: usize,
        out_width: usize,
        y_map: &[NearestEntry],
        x_map: &[NearestEntry],
    ) {
        for (oh, y_entry) in y_map.iter().enumerate() {
            let grad_row = grad_base + oh * out_width;
            if let NearestEntry::Index(ih) = y_entry {
                let in_row = ih * in_width;
                for (ow, x_entry) in x_map.iter().enumerate() {
                    if let NearestEntry::Index(iw) = x_entry {
                        input_grad[in_row + iw] =
                            input_grad[in_row + iw] + grad_data[grad_row + ow];
                    }
                }
            }
        }
    }

    let mut input_grad = vec![T::zero(); in_numel];
    let bc = batch * channels;

    #[cfg(feature = "rayon")]
    if bc * out_hw >= super::PARALLEL_THRESHOLD {
        use rayon::prelude::*;

        input_grad
            .par_chunks_mut(in_hw)
            .enumerate()
            .for_each(|(bc_idx, grad_plane)| {
                let grad_base = bc_idx * out_hw;
                scatter_plane(
                    grad_data, grad_base, grad_plane, in_width, out_width, &y_map, &x_map,
                );
            });
    } else {
        for bc_idx in 0..bc {
            let grad_base = bc_idx * out_hw;
            let in_start = bc_idx * in_hw;
            scatter_plane(
                grad_data,
                grad_base,
                &mut input_grad[in_start..in_start + in_hw],
                in_width,
                out_width,
                &y_map,
                &x_map,
            );
        }
    }
    #[cfg(not(feature = "rayon"))]
    for bc_idx in 0..bc {
        let grad_base = bc_idx * out_hw;
        let in_start = bc_idx * in_hw;
        scatter_plane(
            grad_data,
            grad_base,
            &mut input_grad[in_start..in_start + in_hw],
            in_width,
            out_width,
            &y_map,
            &x_map,
        );
    }

    FlexTensor::new(
        Bytes::from_elems(input_grad),
        Layout::contiguous(Shape::from(vec![batch, channels, in_height, in_width])),
        x.dtype(),
    )
}

/// Bilinear backward: distributes gradients to 4 source positions weighted by bilinear coefficients.
fn interpolate_bilinear_backward_impl<T>(
    x: FlexTensor,
    grad: FlexTensor,
    output_size: [usize; 2],
    options: &InterpolateOptions,
) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
{
    let grad = grad.to_contiguous();
    let grad_data = grad.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let scale_y = options.scale_factor.map(|s| s[0]);
    let scale_x = options.scale_factor.map(|s| s[1]);
    let y_axis_taps = bilinear_axis_taps(
        in_height,
        out_height,
        scale_y,
        options.coordinate_transform,
        0,
    );
    let x_axis_taps = bilinear_axis_taps(
        in_width,
        out_width,
        scale_x,
        options.coordinate_transform,
        1,
    );

    let in_numel = batch * channels * in_height * in_width;
    let mut input_grad = vec![T::zero(); in_numel];

    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;
    let bc = batch * channels;

    let run_plane = |bc_idx: usize, grad_plane: &mut [T]| {
        let out_base = bc_idx * out_hw;

        for (oh, y_tap) in y_axis_taps.iter().enumerate() {
            if let BilinearTap::InBounds {
                low: y_low,
                high: y_high,
                weight: y_weight,
            } = y_tap
            {
                let yw = T::from(*y_weight).unwrap();
                let one = T::one();

                for (ow, x_tap) in x_axis_taps.iter().enumerate() {
                    if let BilinearTap::InBounds {
                        low: x_low,
                        high: x_high,
                        weight: x_weight,
                    } = x_tap
                    {
                        let xw = T::from(*x_weight).unwrap();
                        let grad_val = grad_data[out_base + oh * out_width + ow];

                        grad_plane[y_low * in_width + x_low] = grad_plane[y_low * in_width + x_low]
                            + grad_val * (one - xw) * (one - yw);
                        grad_plane[y_low * in_width + x_high] =
                            grad_plane[y_low * in_width + x_high] + grad_val * xw * (one - yw);
                        grad_plane[y_high * in_width + x_low] =
                            grad_plane[y_high * in_width + x_low] + grad_val * (one - xw) * yw;
                        grad_plane[y_high * in_width + x_high] =
                            grad_plane[y_high * in_width + x_high] + grad_val * xw * yw;
                    }
                }
            }
        }
    };

    #[cfg(feature = "rayon")]
    if bc * out_hw >= super::PARALLEL_THRESHOLD && bc > 1 && in_hw > 0 {
        use rayon::prelude::*;

        input_grad
            .par_chunks_mut(in_hw)
            .enumerate()
            .for_each(|(bc_idx, grad_plane)| run_plane(bc_idx, grad_plane));
    } else {
        for bc_idx in 0..bc {
            let in_start = bc_idx * in_hw;
            run_plane(bc_idx, &mut input_grad[in_start..in_start + in_hw]);
        }
    }
    #[cfg(not(feature = "rayon"))]
    for bc_idx in 0..bc {
        let in_start = bc_idx * in_hw;
        run_plane(bc_idx, &mut input_grad[in_start..in_start + in_hw]);
    }

    FlexTensor::new(
        Bytes::from_elems(input_grad),
        Layout::contiguous(Shape::from(vec![batch, channels, in_height, in_width])),
        x.dtype(),
    )
}

/// Bicubic backward: distributes gradients to 16 source positions weighted by cubic coefficients.
fn interpolate_bicubic_backward_impl<T>(
    x: FlexTensor,
    grad: FlexTensor,
    output_size: [usize; 2],
    options: &InterpolateOptions,
) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
{
    let grad = grad.to_contiguous();
    let grad_data = grad.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let scale_y = options.scale_factor.map(|s| s[0]);
    let scale_x = options.scale_factor.map(|s| s[1]);
    let y_axis_taps = bicubic_axis_taps(
        in_height,
        out_height,
        scale_y,
        options.coordinate_transform,
        options.cubic_options,
        0,
    );
    let x_axis_taps = bicubic_axis_taps(
        in_width,
        out_width,
        scale_x,
        options.coordinate_transform,
        options.cubic_options,
        1,
    );

    let in_numel = batch * channels * in_height * in_width;
    let mut input_grad = vec![T::zero(); in_numel];

    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;
    let bc = batch * channels;

    let run_plane = |bc_idx: usize, grad_plane: &mut [T]| {
        let out_base = bc_idx * out_hw;

        for (oh, y_tap) in y_axis_taps.iter().enumerate() {
            if let BicubicTap::InBounds(y_taps) = y_tap {
                for (ow, x_tap) in x_axis_taps.iter().enumerate() {
                    if let BicubicTap::InBounds(x_taps) = x_tap {
                        let grad_val = <T as num_traits::ToPrimitive>::to_f64(
                            &grad_data[out_base + oh * out_width + ow],
                        )
                        .unwrap_or(0.0);

                        for y_t in y_taps {
                            if y_t.weight == 0.0 {
                                continue;
                            }
                            for x_t in x_taps {
                                if x_t.weight == 0.0 {
                                    continue;
                                }
                                let weight = x_t.weight * y_t.weight * grad_val;
                                grad_plane[y_t.index * in_width + x_t.index] = grad_plane
                                    [y_t.index * in_width + x_t.index]
                                    + T::from(weight).unwrap();
                            }
                        }
                    }
                }
            }
        }
    };

    #[cfg(feature = "rayon")]
    if bc * out_hw >= super::PARALLEL_THRESHOLD && bc > 1 && in_hw > 0 {
        use rayon::prelude::*;

        input_grad
            .par_chunks_mut(in_hw)
            .enumerate()
            .for_each(|(bc_idx, grad_plane)| run_plane(bc_idx, grad_plane));
    } else {
        for bc_idx in 0..bc {
            let in_start = bc_idx * in_hw;
            run_plane(bc_idx, &mut input_grad[in_start..in_start + in_hw]);
        }
    }
    #[cfg(not(feature = "rayon"))]
    for bc_idx in 0..bc {
        let in_start = bc_idx * in_hw;
        run_plane(bc_idx, &mut input_grad[in_start..in_start + in_hw]);
    }

    FlexTensor::new(
        Bytes::from_elems(input_grad),
        Layout::contiguous(Shape::from(vec![batch, channels, in_height, in_width])),
        x.dtype(),
    )
}

// ============================================================================
// Dtype conversion helpers
// ============================================================================

fn convert_bf16_to_f32(x: &FlexTensor) -> FlexTensor {
    let x = x.clone().to_contiguous();
    let input = x.storage::<bf16>();
    let output: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(x.layout().shape().clone()),
        DType::F32,
    )
}

fn convert_f32_to_bf16(x: &FlexTensor) -> FlexTensor {
    let x = x.clone().to_contiguous();
    let input = x.storage::<f32>();
    let output: Vec<bf16> = input.iter().map(|v| bf16::from_f32(*v)).collect();
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(x.layout().shape().clone()),
        DType::BF16,
    )
}

// Tests kept here exercise flex-specific behavior: the typed internal helpers,
// axis-tap precomputation, and dtype dispatch that back the public ops.
// End-to-end interpolate correctness across backends lives in the shared module tests
// under crates/burn-backend-tests/tests/tensor/float/module/{bicubic,bilinear,
// lanczos3,nearest}_interpolate.rs.
#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::ops::InterpolateMode;

    fn make_input_f32(batch: usize, channels: usize, height: usize, width: usize) -> FlexTensor {
        let numel = batch * channels * height * width;
        let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
        FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(vec![batch, channels, height, width])),
            DType::F32,
        )
    }

    fn patterned_data<T: Float>(numel: usize) -> Vec<T> {
        (0..numel)
            .map(|i| {
                let value = ((i * 37 + i * i * 13) % 257) as f64 / 17.0 - 7.0;
                T::from(value).unwrap()
            })
            .collect()
    }

    // Snapshot of the pre-optimization implementation and accumulation order.
    // Update only when intentionally changing the numerical compatibility contract.
    fn bicubic_reference<T>(
        input: &[T],
        shape: [usize; 4],
        output_size: [usize; 2],
        align_corners: bool,
    ) -> Vec<T>
    where
        T: Float + burn_backend::Element + bytemuck::Pod,
    {
        let [batch, channels, in_height, in_width] = shape;
        let [out_height, out_width] = output_size;
        let coord_mode = if align_corners {
            CoordinateTransformMode::AlignCorners
        } else {
            CoordinateTransformMode::HalfPixel
        };
        let in_hw = in_height * in_width;
        let out_hw = out_height * out_width;
        let mut output = vec![T::zero(); batch * channels * out_hw];
        let a = -0.75_f64;

        for b in 0..batch {
            for c in 0..channels {
                let in_base = b * channels * in_hw + c * in_hw;
                let out_base = b * channels * out_hw + c * out_hw;

                for oh in 0..out_height {
                    let y_in = match map_coordinate(oh, in_height, out_height, None, coord_mode, 0)
                    {
                        MappedCoord::InBounds(c) => c,
                        _ => panic!("Expected InBounds"),
                    };
                    let y0 = y_in.floor() as isize;

                    for ow in 0..out_width {
                        let x_in =
                            match map_coordinate(ow, in_width, out_width, None, coord_mode, 1) {
                                MappedCoord::InBounds(c) => c,
                                _ => panic!("Expected InBounds"),
                            };
                        let x0 = x_in.floor() as isize;
                        let mut sum = 0.0_f64;

                        for dy in -1..=2_isize {
                            let y = y0 + dy;
                            let y_index = y.clamp(0, in_height as isize - 1) as usize;
                            let y_weight = cubic_weight((y_in - y0 as f64) - dy as f64, a);

                            for dx in -1..=2_isize {
                                let x = x0 + dx;
                                let x_index = x.clamp(0, in_width as isize - 1) as usize;
                                let x_weight = cubic_weight((x_in - x0 as f64) - dx as f64, a);
                                let value = input[in_base + y_index * in_width + x_index];
                                let value =
                                    <T as num_traits::ToPrimitive>::to_f64(&value).unwrap_or(0.0);
                                sum += value * x_weight * y_weight;
                            }
                        }

                        output[out_base + oh * out_width + ow] = T::from(sum).unwrap();
                    }
                }
            }
        }

        output
    }

    // Snapshot of the pre-optimization implementation and accumulation order.
    // Update only when intentionally changing the numerical compatibility contract.
    fn lanczos3_reference<T>(
        input: &[T],
        shape: [usize; 4],
        output_size: [usize; 2],
        align_corners: bool,
    ) -> Vec<T>
    where
        T: Float + burn_backend::Element + bytemuck::Pod,
    {
        let [batch, channels, in_height, in_width] = shape;
        let [out_height, out_width] = output_size;
        let coord_mode = if align_corners {
            CoordinateTransformMode::AlignCorners
        } else {
            CoordinateTransformMode::HalfPixel
        };
        let in_hw = in_height * in_width;
        let out_hw = out_height * out_width;
        let max_h = in_height as isize - 1;
        let max_w = in_width as isize - 1;
        let mut output = vec![T::zero(); batch * channels * out_hw];

        for b in 0..batch {
            for c in 0..channels {
                let in_base = b * channels * in_hw + c * in_hw;
                let out_base = b * channels * out_hw + c * out_hw;

                for oh in 0..out_height {
                    let y_in = match map_coordinate(oh, in_height, out_height, None, coord_mode, 0)
                    {
                        MappedCoord::InBounds(c) => c,
                        _ => panic!("Expected InBounds"),
                    };
                    let y0 = y_in.floor();

                    for ow in 0..out_width {
                        let x_in =
                            match map_coordinate(ow, in_width, out_width, None, coord_mode, 1) {
                                MappedCoord::InBounds(c) => c,
                                _ => panic!("Expected InBounds"),
                            };
                        let x0 = x_in.floor();
                        let mut result = 0.0_f64;
                        let mut weight_sum = 0.0_f64;

                        for ky in -2..=3_isize {
                            let y_index = y0 as isize + ky;
                            if y_index < 0 || y_index > max_h {
                                continue;
                            }
                            let y_weight = lanczos3_weight(y_in - (y0 + ky as f64));

                            for kx in -2..=3_isize {
                                let x_index = x0 as isize + kx;
                                if x_index < 0 || x_index > max_w {
                                    continue;
                                }
                                let x_weight = lanczos3_weight(x_in - (x0 + kx as f64));
                                let weight = y_weight * x_weight;
                                let value =
                                    input[in_base + y_index as usize * in_width + x_index as usize];
                                let value =
                                    <T as num_traits::ToPrimitive>::to_f64(&value).unwrap_or(0.0);
                                result += value * weight;
                                weight_sum += weight;
                            }
                        }

                        if weight_sum != 0.0 {
                            result /= weight_sum;
                        }
                        output[out_base + oh * out_width + ow] = T::from(result).unwrap();
                    }
                }
            }
        }

        output
    }

    fn assert_bitwise_equal<T: bytemuck::Pod>(
        actual: &[T],
        expected: &[T],
        shape: [usize; 4],
        output_size: [usize; 2],
        align_corners: bool,
    ) {
        assert_eq!(
            bytemuck::cast_slice::<T, u8>(actual),
            bytemuck::cast_slice::<T, u8>(expected),
            "shape={shape:?}, output_size={output_size:?}, align_corners={align_corners}"
        );
    }

    fn assert_bicubic_matches_reference<T>(
        shape: [usize; 4],
        output_size: [usize; 2],
        align_corners: bool,
    ) where
        T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
    {
        let numel = shape.iter().product();
        let data = patterned_data::<T>(numel);
        let expected = bicubic_reference(&data, shape, output_size, align_corners);
        let input = FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            <T as burn_backend::Element>::dtype(),
        );
        let options =
            InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(align_corners);
        let actual = interpolate_bicubic_impl::<T>(input, output_size, &options);
        assert_bitwise_equal(
            actual.storage::<T>(),
            &expected,
            shape,
            output_size,
            align_corners,
        );
    }

    fn assert_lanczos3_matches_reference<T>(
        shape: [usize; 4],
        output_size: [usize; 2],
        align_corners: bool,
    ) where
        T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
    {
        let numel = shape.iter().product();
        let data = patterned_data::<T>(numel);
        let expected = lanczos3_reference(&data, shape, output_size, align_corners);
        let input = FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            <T as burn_backend::Element>::dtype(),
        );
        let options =
            InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(align_corners);
        let actual = interpolate_lanczos3_impl::<T>(input, output_size, &options);
        assert_bitwise_equal(
            actual.storage::<T>(),
            &expected,
            shape,
            output_size,
            align_corners,
        );
    }

    fn assert_bicubic_bf16_matches_reference(
        shape: [usize; 4],
        output_size: [usize; 2],
        align_corners: bool,
    ) {
        let numel = shape.iter().product();
        let data: Vec<bf16> = patterned_data::<f32>(numel)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let data_f32: Vec<f32> = data.iter().map(|value| value.to_f32()).collect();
        let expected: Vec<bf16> = bicubic_reference(&data_f32, shape, output_size, align_corners)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let input = FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::BF16,
        );
        let options =
            InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(align_corners);
        let actual = interpolate_bicubic_bf16(input, output_size, &options);
        assert_bitwise_equal(
            actual.storage::<bf16>(),
            &expected,
            shape,
            output_size,
            align_corners,
        );
    }

    fn assert_lanczos3_bf16_matches_reference(
        shape: [usize; 4],
        output_size: [usize; 2],
        align_corners: bool,
    ) {
        let numel = shape.iter().product();
        let data: Vec<bf16> = patterned_data::<f32>(numel)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let data_f32: Vec<f32> = data.iter().map(|value| value.to_f32()).collect();
        let expected: Vec<bf16> = lanczos3_reference(&data_f32, shape, output_size, align_corners)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let input = FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::BF16,
        );
        let options =
            InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(align_corners);
        let actual = interpolate_lanczos3_bf16(input, output_size, &options);
        assert_bitwise_equal(
            actual.storage::<bf16>(),
            &expected,
            shape,
            output_size,
            align_corners,
        );
    }

    const EQUIVALENCE_CASES: [([usize; 4], [usize; 2]); 8] = [
        ([1, 1, 1, 1], [1, 1]),
        ([1, 1, 1, 5], [3, 9]),
        ([1, 1, 5, 1], [9, 3]),
        ([1, 2, 4, 5], [7, 8]),
        ([2, 3, 9, 7], [3, 4]),
        ([1, 1, 7, 5], [1, 1]),
        ([1, 7, 4, 5], [7, 8]),
        ([1, 8, 4, 5], [7, 8]),
    ];

    #[test]
    fn test_bicubic_axis_taps_match_direct_computation() {
        let a = -0.75_f64;
        let cubic_options = CubicOptions {
            a: a as f32,
            exclude_outside: false,
        };
        for in_size in 1..=8 {
            for out_size in 1..=9 {
                for align_corners in [false, true] {
                    let coord_mode = if align_corners {
                        CoordinateTransformMode::AlignCorners
                    } else {
                        CoordinateTransformMode::HalfPixel
                    };
                    let table =
                        bicubic_axis_taps(in_size, out_size, None, coord_mode, cubic_options, 0);

                    for (out_coord, tap) in table.iter().enumerate() {
                        let coord =
                            match map_coordinate(out_coord, in_size, out_size, None, coord_mode, 0)
                            {
                                MappedCoord::InBounds(c) => c,
                                _ => panic!("Expected InBounds"),
                            };
                        let base = coord.floor() as isize;

                        let axis_taps = match tap {
                            BicubicTap::InBounds(t) => t,
                            _ => panic!("Expected InBounds"),
                        };

                        for (tap, offset) in axis_taps.iter().zip(-1..=2_isize) {
                            let index = (base + offset).clamp(0, in_size as isize - 1) as usize;
                            let weight = cubic_weight((coord - base as f64) - offset as f64, a);
                            assert_eq!(tap.index, index);
                            assert_eq!(tap.weight.to_bits(), weight.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_lanczos3_axis_taps_match_direct_computation() {
        for in_size in 1..=8 {
            for out_size in 1..=9 {
                for align_corners in [false, true] {
                    let coord_mode = if align_corners {
                        CoordinateTransformMode::AlignCorners
                    } else {
                        CoordinateTransformMode::HalfPixel
                    };
                    let table = lanczos3_axis_taps(in_size, out_size, None, coord_mode, 0);

                    for (out_coord, tap) in table.iter().enumerate() {
                        let coord =
                            match map_coordinate(out_coord, in_size, out_size, None, coord_mode, 0)
                            {
                                MappedCoord::InBounds(c) => c,
                                _ => panic!("Expected InBounds"),
                            };
                        let base = coord.floor();
                        let expected: Vec<AxisTap> = (-2..=3_isize)
                            .filter_map(|offset| {
                                let index = base as isize + offset;
                                (index >= 0 && index < in_size as isize).then(|| AxisTap {
                                    index: index as usize,
                                    weight: lanczos3_weight(coord - (base + offset as f64)),
                                })
                            })
                            .collect();

                        let axis_taps = match tap {
                            Lanczos3Tap::InBounds(t) => t,
                            _ => panic!("Expected InBounds"),
                        };

                        assert_eq!(axis_taps.len, expected.len());
                        for (tap, expected) in axis_taps.as_slice().iter().zip(expected) {
                            assert_eq!(tap.index, expected.index);
                            assert_eq!(tap.weight.to_bits(), expected.weight.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_bicubic_matches_reference_for_all_dtypes() {
        for (shape, output_size) in EQUIVALENCE_CASES {
            for align_corners in [false, true] {
                assert_bicubic_matches_reference::<f32>(shape, output_size, align_corners);
                assert_bicubic_matches_reference::<f64>(shape, output_size, align_corners);
                assert_bicubic_matches_reference::<f16>(shape, output_size, align_corners);
                assert_bicubic_bf16_matches_reference(shape, output_size, align_corners);
            }
        }
    }

    #[test]
    fn test_lanczos3_matches_reference_for_all_dtypes() {
        for (shape, output_size) in EQUIVALENCE_CASES {
            for align_corners in [false, true] {
                assert_lanczos3_matches_reference::<f32>(shape, output_size, align_corners);
                assert_lanczos3_matches_reference::<f64>(shape, output_size, align_corners);
                assert_lanczos3_matches_reference::<f16>(shape, output_size, align_corners);
                assert_lanczos3_bf16_matches_reference(shape, output_size, align_corners);
            }
        }
    }

    #[test]
    fn test_lanczos3_skips_out_of_bounds_taps() {
        let mut data: Vec<f32> = (0..8).map(|value| value as f32).collect();
        data[0] = f32::NAN;
        let expected = lanczos3_reference(&data, [1, 1, 1, 8], [1, 8], true);
        let input = FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(vec![1, 1, 1, 8])),
            DType::F32,
        );
        let options = InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(true);
        let actual = interpolate_lanczos3_f32(input, [1, 8], &options);
        let actual = actual.storage::<f32>();

        assert!(actual[7].is_finite());
        assert_eq!(actual[7].to_bits(), expected[7].to_bits());
    }

    #[test]
    fn test_nearest_upsample_2x() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(vec![1, 1, 2, 2])),
            DType::F32,
        );

        let options = InterpolateOptions::new(InterpolateMode::Nearest);
        let result = interpolate_nearest_f32(x, [4, 4], &options);
        let output = result.storage::<f32>();

        assert_eq!(output.len(), 16);
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 2.0);
        assert_eq!(output[3], 2.0);
    }

    #[test]
    fn test_bilinear_upsample_2x() {
        let data = vec![0.0f32, 1.0, 1.0, 0.0];
        let x = FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(vec![1, 1, 2, 2])),
            DType::F32,
        );

        let options = InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(true);
        let result = interpolate_bilinear_f32(x, [4, 4], &options);
        let output = result.storage::<f32>();

        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-5);
        assert!((output[12] - 1.0).abs() < 1e-5);
        assert!((output[15] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_bicubic_basic() {
        let x = make_input_f32(1, 1, 4, 4);
        let options = InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(true);
        let result = interpolate_bicubic_f32(x, [8, 8], &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 8, 8]);
    }

    #[test]
    fn test_downsample() {
        let x = make_input_f32(1, 1, 4, 4);
        let options = InterpolateOptions::new(InterpolateMode::Nearest);
        let result = interpolate_nearest_f32(x, [2, 2], &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_nearest_backward() {
        let x = make_input_f32(1, 1, 2, 2);
        let grad = FlexTensor::new(
            Bytes::from_elems(vec![1.0f32; 16]),
            Layout::contiguous(Shape::from(vec![1, 1, 4, 4])),
            DType::F32,
        );

        let options = InterpolateOptions::new(InterpolateMode::Nearest);
        let result = interpolate_nearest_backward_f32(x, grad, [4, 4], &options);
        let output = result.storage::<f32>();

        assert_eq!(output.len(), 4);
        assert!((output[0] - 4.0).abs() < 1e-5);
    }
}
