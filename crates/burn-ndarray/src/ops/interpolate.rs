use burn_backend::ElementConversion;
use ndarray::{Array4, ArrayBase, DataOwned};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{FloatNdArrayElement, ShapeOps, SharedArray, UnsafeSharedRef, iter_range_par, run_par};

pub(crate) fn nearest_interpolate<E: FloatNdArrayElement>(
    x: SharedArray<E>,
    output_size: [usize; 2],
) -> SharedArray<E> {
    let x = x.into_dimensionality::<ndarray::Ix4>().unwrap();

    let (batch_size, channels, in_height, in_width) = x.dim();
    let [out_height, out_width] = output_size;

    let y_ratio = (in_height as f64) / (out_height as f64);
    let x_ratio = (in_width as f64) / (out_width as f64);

    let out_element_num = batch_size * channels * out_height * out_width;
    let strides = (
        channels * out_height * out_width,
        out_height * out_width,
        out_width,
    );

    let mut output = Array4::zeros((batch_size, channels, out_height, out_width));
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_range_par!(0, out_element_num).for_each(|id| {
            let (b, c, h, w) = (
                id / strides.0,
                id % strides.0 / strides.1,
                id % strides.1 / strides.2,
                id % strides.2,
            );

            let y_in = (y_ratio * h as f64).floor() as usize;
            let x_in = (x_ratio * w as f64).floor() as usize;

            unsafe {
                let output = unsafe_shared_out.get();
                output[(b, c, h, w)] = x[(b, c, y_in, x_in)];
            }
        });
    });

    output.into_dyn().into_shared()
}

pub(crate) fn nearest_interpolate_backward<E: FloatNdArrayElement>(
    x: SharedArray<E>,
    grad: SharedArray<E>,
    output_size: [usize; 2],
) -> SharedArray<E> {
    let [batch_size, channels, input_height, input_width] = x.shape().dims();
    let [output_height, output_width] = output_size;

    let mut output_grad =
        Array4::from_elem((batch_size, channels, input_height, input_width), 0.elem());
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output_grad);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let output_grad = unsafe_shared_out.get();

            for oh in 0..output_height {
                for ow in 0..output_width {
                    let ih = start_index(oh, output_height, input_height);
                    let iw = start_index(ow, output_width, input_width);

                    output_grad[[b, c, ih, iw]] += grad[[b, c, oh, ow]]
                }
            }
        })
    });

    output_grad.into_dyn().into_shared()
}

fn start_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    ((output_size_index as f32 * input_size as f32) / output_size as f32).floor() as usize
}

// clamp ceil(frac) to stay within bounds in case of floating-point imprecision
pub(crate) fn ceil_clamp(frac: f64, max: usize) -> f64 {
    frac.ceil().min(max as f64)
}

pub(crate) fn bilinear_interpolate<E: FloatNdArrayElement>(
    x: SharedArray<E>,
    output_size: [usize; 2],
    align_corners: bool,
) -> SharedArray<E> {
    let x = x.into_dimensionality::<ndarray::Ix4>().unwrap();

    let (batch_size, channels, in_height, in_width) = x.dim();
    let [out_height, out_width] = output_size;

    let out_element_num = batch_size * channels * out_height * out_width;
    let strides = (
        channels * out_height * out_width,
        out_height * out_width,
        out_width,
    );

    let mut output = Array4::zeros((batch_size, channels, out_height, out_width));
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_range_par!(0, out_element_num).for_each(|id| {
            let (b, c, h, w) = (
                id / strides.0,
                id % strides.0 / strides.1,
                id % strides.1 / strides.2,
                id % strides.2,
            );

            let (y_frac, x_frac) = if align_corners {
                let y_ratio = ((in_height - 1) as f64) / (core::cmp::max(out_height - 1, 1) as f64);
                let x_ratio = ((in_width - 1) as f64) / (core::cmp::max(out_width - 1, 1) as f64);
                (y_ratio * h as f64, x_ratio * w as f64)
            } else {
                let y_frac = (h as f64 + 0.5) * (in_height as f64 / out_height as f64) - 0.5;
                let x_frac = (w as f64 + 0.5) * (in_width as f64 / out_width as f64) - 0.5;
                (
                    y_frac.clamp(0.0, (in_height - 1) as f64),
                    x_frac.clamp(0.0, (in_width - 1) as f64),
                )
            };
            let val =
                bilinear_interpolate_single(&x, b, c, x_frac, y_frac, in_width - 1, in_height - 1);

            unsafe {
                let output = unsafe_shared_out.get();
                output[(b, c, h, w)] = val.elem();
            }
        });
    });

    output.into_dyn().into_shared()
}

pub(crate) fn bicubic_interpolate<E: FloatNdArrayElement>(
    x: SharedArray<E>,
    output_size: [usize; 2],
    align_corners: bool,
) -> SharedArray<E> {
    fn cubic_interp1d(x0: f64, x1: f64, x2: f64, x3: f64, t: f64) -> f64 {
        fn cubic_convolution1(x: f64, a: f64) -> f64 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        }

        fn cubic_convolution2(x: f64, a: f64) -> f64 {
            ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
        }

        let coeffs = [
            cubic_convolution2(t + 1.0, -0.75),
            cubic_convolution1(t, -0.75),
            cubic_convolution1(1.0 - t, -0.75),
            cubic_convolution2(2.0 - t, -0.75),
        ];

        x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3]
    }

    let x = x.into_dimensionality::<ndarray::Ix4>().unwrap();

    let (batch_size, channels, in_height, in_width) = x.dim();
    let [out_height, out_width] = output_size;

    let out_element_num = batch_size * channels * out_height * out_width;
    let strides = (
        channels * out_height * out_width,
        out_height * out_width,
        out_width,
    );

    let mut output = Array4::zeros((batch_size, channels, out_height, out_width));
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_range_par!(0, out_element_num).for_each(|id| {
            let (b, c, h, w) = (
                id / strides.0,
                id % strides.0 / strides.1,
                id % strides.1 / strides.2,
                id % strides.2,
            );

            let (y_frac, x_frac) = if align_corners {
                let y_ratio = ((in_height - 1) as f64) / (core::cmp::max(out_height - 1, 1) as f64);
                let x_ratio = ((in_width - 1) as f64) / (core::cmp::max(out_width - 1, 1) as f64);
                (y_ratio * h as f64, x_ratio * w as f64)
            } else {
                let y_frac = (h as f64 + 0.5) * (in_height as f64 / out_height as f64) - 0.5;
                let x_frac = (w as f64 + 0.5) * (in_width as f64 / out_width as f64) - 0.5;
                (y_frac, x_frac)
            };
            let y0 = y_frac.floor();
            let yw = y_frac - y0;
            let y_in = y0 as isize;

            let x0 = x_frac.floor();
            let xw = x_frac - x0;
            let x_in = x0 as isize;

            let max_h = (in_height - 1) as isize;
            let max_w = (in_width - 1) as isize;

            let ys_in = [
                (y_in - 1).clamp(0, max_h) as usize,
                y_in.clamp(0, max_h) as usize,
                (y_in + 1).clamp(0, max_h) as usize,
                (y_in + 2).clamp(0, max_h) as usize,
            ];

            let xs_in = [
                (x_in - 1).clamp(0, max_w) as usize,
                x_in.clamp(0, max_w) as usize,
                (x_in + 1).clamp(0, max_w) as usize,
                (x_in + 2).clamp(0, max_w) as usize,
            ];

            let coefficients = ys_in.map(|y| {
                cubic_interp1d(
                    x[(b, c, y, xs_in[0])].elem(),
                    x[(b, c, y, xs_in[1])].elem(),
                    x[(b, c, y, xs_in[2])].elem(),
                    x[(b, c, y, xs_in[3])].elem(),
                    xw,
                )
            });

            let result = cubic_interp1d(
                coefficients[0],
                coefficients[1],
                coefficients[2],
                coefficients[3],
                yw,
            )
            .elem();

            unsafe {
                let output = unsafe_shared_out.get();
                output[(b, c, h, w)] = result;
            }
        });
    });

    output.into_dyn().into_shared()
}

/// Sample an element of the source array with bilinear interpolation
///
/// * `source` - The tensor to read from. Has shape (batch_size, channels, height, width)
/// * `b` - The batch to read from
/// * `c` - The channel to read from
/// * `x` - The x position to read in the array
/// * `y` - The y position to read in the array
/// * `x_max` - The max x position (inclusive)
/// * `y_max` - The max y position (inclusive)
///
/// # Returns
///
/// The interpolated value read from the array
pub(crate) fn bilinear_interpolate_single<E, S>(
    source: &ArrayBase<S, ndarray::Dim<[usize; 4]>>,
    b: usize,
    c: usize,
    x: f64,
    y: f64,
    x_max: usize,
    y_max: usize,
) -> f64
where
    E: FloatNdArrayElement,
    S: DataOwned<Elem = E>,
{
    let y0 = y.floor();
    let y1 = ceil_clamp(y, y_max);
    let yw = y - y0;

    let x0 = x.floor();
    let x1 = ceil_clamp(x, x_max);
    let xw = x - x0;

    let (x0, x1, y0, y1) = (x0 as usize, x1 as usize, y0 as usize, y1 as usize);

    let p_a = source[(b, c, y0, x0)].elem::<f64>() * (1.0 - xw) * (1.0 - yw);
    let p_b = source[(b, c, y0, x1)].elem::<f64>() * xw * (1.0 - yw);
    let p_c = source[(b, c, y1, x0)].elem::<f64>() * (1.0 - xw) * yw;
    let p_d = source[(b, c, y1, x1)].elem::<f64>() * xw * yw;

    p_a + p_b + p_c + p_d
}
