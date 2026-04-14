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
                        let xi = px.round() as i64;
                        let yi = py.round() as i64;
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;
    use burn_backend::ops::GridSampleOptions;

    fn make_grid_f32(data: &[f32], shape: [usize; 4]) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data.to_vec(), shape.to_vec()))
    }

    #[test]
    fn test_nearest_basic() {
        // 1x1x2x2 input: [[1, 2], [3, 4]]
        let input = make_grid_f32(&[1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]);
        // Grid pointing to corners: (-1,-1)=top-left, (1,1)=bottom-right
        let grid = make_grid_f32(&[-1.0, -1.0, 1.0, 1.0], [1, 1, 2, 2]);

        let options = GridSampleOptions {
            mode: InterpolateMode::Nearest,
            padding_mode: GridSamplePaddingMode::Zeros,
            align_corners: true,
        };
        let result = grid_sample_2d(input, grid, options);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 4.0]); // top-left, bottom-right
    }

    #[test]
    fn test_nearest_vs_bilinear() {
        // 1x1x3x3 input with distinct values
        let input = make_grid_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1, 1, 3, 3]);
        // Grid at center (0,0) with align_corners=true maps to (1,1) = 5.0
        let grid = make_grid_f32(&[0.0, 0.0], [1, 1, 1, 2]);

        let nearest_opts = GridSampleOptions {
            mode: InterpolateMode::Nearest,
            padding_mode: GridSamplePaddingMode::Zeros,
            align_corners: true,
        };
        let bilinear_opts = GridSampleOptions {
            mode: InterpolateMode::Bilinear,
            padding_mode: GridSamplePaddingMode::Zeros,
            align_corners: true,
        };

        let nearest: Vec<f32> = grid_sample_2d(input.clone(), grid.clone(), nearest_opts)
            .into_data()
            .to_vec()
            .unwrap();
        let bilinear: Vec<f32> = grid_sample_2d(input, grid, bilinear_opts)
            .into_data()
            .to_vec()
            .unwrap();
        // At exact center, both should return 5.0
        assert_eq!(nearest, vec![5.0]);
        assert_eq!(bilinear, vec![5.0]);
    }

    #[test]
    fn test_nearest_off_center() {
        // 1x1x2x2 input: [[10, 20], [30, 40]]
        let input = make_grid_f32(&[10.0, 20.0, 30.0, 40.0], [1, 1, 2, 2]);
        // Grid slightly left of center: x=-0.1 maps closer to left column
        let grid = make_grid_f32(&[-0.1, -0.1], [1, 1, 1, 2]);

        let options = GridSampleOptions {
            mode: InterpolateMode::Nearest,
            padding_mode: GridSamplePaddingMode::Zeros,
            align_corners: true,
        };
        let result: Vec<f32> = grid_sample_2d(input, grid, options)
            .into_data()
            .to_vec()
            .unwrap();
        // (-0.1, -0.1) with align_corners=true on 2x2: px = 0.45, py = 0.45
        // Nearest rounds to (0, 0) = 10.0
        assert_eq!(result, vec![10.0]);
    }

    #[test]
    fn test_nearest_border_padding() {
        // 1x1x2x2 input
        let input = make_grid_f32(&[1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]);
        // Grid outside bounds
        let grid = make_grid_f32(&[3.0, 3.0], [1, 1, 1, 2]);

        let options = GridSampleOptions {
            mode: InterpolateMode::Nearest,
            padding_mode: GridSamplePaddingMode::Border,
            align_corners: true,
        };
        let result: Vec<f32> = grid_sample_2d(input, grid, options)
            .into_data()
            .to_vec()
            .unwrap();
        // Border padding clamps to edge: bottom-right = 4.0
        assert_eq!(result, vec![4.0]);
    }

    #[test]
    fn test_bilinear_basic() {
        // Verify existing bilinear still works
        let input = make_grid_f32(&[1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]);
        let grid = make_grid_f32(&[0.0, 0.0], [1, 1, 1, 2]);

        let options = GridSampleOptions {
            mode: InterpolateMode::Bilinear,
            padding_mode: GridSamplePaddingMode::Zeros,
            align_corners: true,
        };
        let result: Vec<f32> = grid_sample_2d(input, grid, options)
            .into_data()
            .to_vec()
            .unwrap();
        // Center of 2x2 with align_corners: average of all 4 = 2.5
        assert_eq!(result, vec![2.5]);
    }
}
