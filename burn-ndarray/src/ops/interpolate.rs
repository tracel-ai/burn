use burn_tensor::ElementConversion;
use ndarray::Array4;

use crate::{iter_range_par, run_par, FloatNdArrayElement, NdArrayTensor};

// FIXME: this function gives different result with other backends
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

    let mut output_data = Vec::with_capacity(out_element_num);

    run_par!(|| {
        iter_range_par!(0, out_element_num)
            .map(|id| {
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

                (p_a + p_b + p_c + p_d).elem()
            })
            .collect_into_vec(&mut output_data);
    });

    let output =
        Array4::from_shape_vec((batch_size, channels, out_height, out_width), output_data).unwrap();

    NdArrayTensor::new(output.into_dyn().into_shared())
}
