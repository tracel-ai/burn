use burn_tensor::ElementConversion;
use burn_tensor::ops::InterpolateMode;

use ndarray::Array4;

use crate::{FloatNdArrayElement, UnsafeSharedRef, iter_range_par, run_par};
use crate::SharedArray;

/// Sample a tensor using grid-based sampling with bilinear interpolation.
///
/// Uses ONNX semantics: align_corners=false and padding_mode=zeros.
/// Grid values in [-1, 1] map to pixel coordinates as follows:
/// - align_corners=false: x_pixel = (x_norm + 1) * width / 2 - 0.5
///   This maps -1 -> -0.5 and 1 -> width - 0.5
///
/// Out-of-bounds coordinates return 0 (zeros padding).
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

            // Convert normalized grid coordinates [-1, 1] to pixel coordinates
            // Using align_corners=false semantics:
            // x_pixel = (x_norm + 1) * width / 2 - 0.5
            // This maps -1 -> -0.5 and 1 -> width - 0.5
            let px = (sample_x + 1.0) * (width_in as f64) / 2.0 - 0.5;
            let py = (sample_y + 1.0) * (height_in as f64) / 2.0 - 0.5;

            // Bilinear interpolation with zeros padding
            let val = bilinear_interpolate_zeros(&tensor, b, c, px, py, width_in, height_in);

            unsafe {
                let output = unsafe_shared_out.get();
                output[(b, c, y, x)] = val.elem();
            }
        });
    });

    output.into_dyn().into_shared()
}

/// Bilinear interpolation at a point with zeros padding.
///
/// Returns 0 for any out-of-bounds sample contributions (including inf/nan).
fn bilinear_interpolate_zeros<E, S>(
    source: &ndarray::ArrayBase<S, ndarray::Dim<[usize; 4]>>,
    b: usize,
    c: usize,
    x: f64,
    y: f64,
    width: usize,
    height: usize,
) -> f64
where
    E: FloatNdArrayElement,
    S: ndarray::Data<Elem = E>,
{
    // Handle inf/nan coordinates - return 0 (zeros padding for out-of-bounds)
    if !x.is_finite() || !y.is_finite() {
        return 0.0;
    }

    // Get the four corner indices (safe now that we've handled inf/nan)
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0.saturating_add(1);
    let y1 = y0.saturating_add(1);

    // Compute interpolation weights (fractional part)
    let x_frac = x - x.floor();
    let y_frac = y - y.floor();

    // Helper to safely read a value, returning 0 for out-of-bounds (zeros padding)
    let read_safe = |xi: i64, yi: i64| -> f64 {
        if xi >= 0 && xi < width as i64 && yi >= 0 && yi < height as i64 {
            source[(b, c, yi as usize, xi as usize)].elem::<f64>()
        } else {
            0.0
        }
    };

    // Read the four corners, with zeros padding for out-of-bounds
    let v00 = read_safe(x0, y0);
    let v01 = read_safe(x0, y1);
    let v10 = read_safe(x1, y0);
    let v11 = read_safe(x1, y1);

    // Bilinear interpolation weights
    let w00 = (1.0 - x_frac) * (1.0 - y_frac);
    let w01 = (1.0 - x_frac) * y_frac;
    let w10 = x_frac * (1.0 - y_frac);
    let w11 = x_frac * y_frac;

    v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
}
