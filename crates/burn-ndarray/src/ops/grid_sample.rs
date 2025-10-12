use burn_tensor::ElementConversion;
use burn_tensor::ops::InterpolateMode;

use ndarray::Array4;

use crate::{FloatNdArrayElement, UnsafeSharedRef, iter_range_par, run_par};
use crate::{SharedArray, ops::interpolate::bilinear_interpolate_single};

/// Sample a tensor
pub(crate) fn grid_sample_2d<E: FloatNdArrayElement>(
    tensor: SharedArray<E>,
    grid: SharedArray<E>,
    method: InterpolateMode,
) -> SharedArray<E> {
    match method {
        InterpolateMode::Bilinear => (),
        _ => todo!("Unimplemented"),
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

            let sample_x = sample_x.clamp(-1.0, 1.0);
            let sample_y = sample_y.clamp(-1.0, 1.0);

            let x_max = width_in - 1;
            let y_max = height_in - 1;
            let x_max_half = x_max as f64 / 2.0;
            let y_max_half = y_max as f64 / 2.0;

            // Scale from (-1, 1) to (0, dim_max)
            let sample_x = sample_x * x_max_half + x_max_half;
            let sample_y = sample_y * y_max_half + y_max_half;

            let val = bilinear_interpolate_single(&tensor, b, c, sample_x, sample_y, x_max, y_max);

            unsafe {
                let output = unsafe_shared_out.get();
                output[(b, c, y, x)] = val.elem();
            }
        });
    });

    output.into_dyn().into_shared()
}
