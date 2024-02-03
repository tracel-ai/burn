use burn_tensor::ElementConversion;
use ndarray::Array4;
#[cfg(not(feature = "std"))]
use num_traits::Float;

use crate::{iter_range_par, run_par, FloatNdArrayElement, NdArrayTensor, UnsafeSharedRef};

pub(crate) fn nearest_interpolate<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    output_size: [usize; 2],
) -> NdArrayTensor<E, 4> {
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

pub(crate) fn bilinear_interpolate<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    output_size: [usize; 2],
) -> NdArrayTensor<E, 4> {
    let x = x.array.into_dimensionality::<ndarray::Ix4>().unwrap();

    let (batch_size, channels, in_height, in_width) = x.dim();
    let [out_height, out_width] = output_size;

    let y_ratio = ((in_height - 1) as f64) / ((out_height - 1) as f64);
    let x_ratio = ((in_width - 1) as f64) / ((out_width - 1) as f64);

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
