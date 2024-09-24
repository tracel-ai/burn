use burn_common::{iter_range_par, run_par};
use burn_tensor::ElementConversion;
use ndarray::Array4;
#[cfg(not(feature = "std"))]
use num_traits::Float;

use crate::{FloatNdArrayElement, NdArrayTensor, UnsafeSharedRef};

pub(crate) fn nearest_interpolate<E: FloatNdArrayElement>(
    x: NdArrayTensor<E>,
    output_size: [usize; 2],
) -> NdArrayTensor<E> {
    let x = x.array.into_dimensionality::<ndarray::Ix4>().unwrap();

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

    NdArrayTensor::new(output.into_dyn().into_shared())
}

pub(crate) fn nearest_interpolate_backward<E: FloatNdArrayElement>(
    x: NdArrayTensor<E>,
    grad: NdArrayTensor<E>,
    output_size: [usize; 2],
) -> NdArrayTensor<E> {
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

                    output_grad[[b, c, ih, iw]] += grad.array[[b, c, oh, ow]]
                }
            }
        })
    });

    NdArrayTensor::new(output_grad.into_dyn().into_shared())
}

fn start_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    ((output_size_index as f32 * input_size as f32) / output_size as f32).floor() as usize
}

pub(crate) fn bilinear_interpolate<E: FloatNdArrayElement>(
    x: NdArrayTensor<E>,
    output_size: [usize; 2],
) -> NdArrayTensor<E> {
    let x = x.array.into_dimensionality::<ndarray::Ix4>().unwrap();

    let (batch_size, channels, in_height, in_width) = x.dim();
    let [out_height, out_width] = output_size;

    let y_ratio = ((in_height - 1) as f64) / (core::cmp::max(out_height - 1, 1) as f64);
    let x_ratio = ((in_width - 1) as f64) / (core::cmp::max(out_width - 1, 1) as f64);

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

            // We convert everything to `f64` for calculations and then back to `E` at the end.
            let y_frac = y_ratio * h as f64;
            let y0 = y_frac.floor();
            let y1 = y_frac.ceil();
            let yw = y_frac - y0;

            let x_frac = x_ratio * w as f64;
            let x0 = x_frac.floor();
            let x1 = x_frac.ceil();
            let xw = x_frac - x0;

            let (x0, x1, y0, y1) = (x0 as usize, x1 as usize, y0 as usize, y1 as usize);

            let p_a = x[(b, c, y0, x0)].elem::<f64>() * (1.0 - xw) * (1.0 - yw);
            let p_b = x[(b, c, y0, x1)].elem::<f64>() * xw * (1.0 - yw);
            let p_c = x[(b, c, y1, x0)].elem::<f64>() * (1.0 - xw) * yw;
            let p_d = x[(b, c, y1, x1)].elem::<f64>() * xw * yw;

            unsafe {
                let output = unsafe_shared_out.get();
                output[(b, c, h, w)] = (p_a + p_b + p_c + p_d).elem();
            }
        });
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}

pub(crate) fn bicubic_interpolate<E: FloatNdArrayElement>(
    x: NdArrayTensor<E>,
    output_size: [usize; 2],
) -> NdArrayTensor<E> {
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

    let x = x.array.into_dimensionality::<ndarray::Ix4>().unwrap();

    let (batch_size, channels, in_height, in_width) = x.dim();
    let [out_height, out_width] = output_size;

    let y_ratio = ((in_height - 1) as f64) / (core::cmp::max(out_height - 1, 1) as f64);
    let x_ratio = ((in_width - 1) as f64) / (core::cmp::max(out_width - 1, 1) as f64);

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

            let y_frac = y_ratio * h as f64;
            let y0 = y_frac.floor();
            let yw = y_frac - y0;
            let y_in = y0 as usize;

            let x_frac = x_ratio * w as f64;
            let x0 = x_frac.floor();
            let xw = x_frac - x0;
            let x_in = x0 as usize;

            let ys_in = [
                if y_in == 0 { 0 } else { y_in - 1 },
                y_in,
                y_in + 1,
                y_in + 2,
            ]
            .map(|y| y.min(in_height - 1));

            let xs_in = [
                if x_in == 0 { 0 } else { x_in - 1 },
                x_in,
                x_in + 1,
                x_in + 2,
            ]
            .map(|x| x.min(in_width - 1));

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

    NdArrayTensor::new(output.into_dyn().into_shared())
}
