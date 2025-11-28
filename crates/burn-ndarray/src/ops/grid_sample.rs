use burn_tensor::ElementConversion;
use burn_tensor::ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode};

use ndarray::Array4;

use crate::SharedArray;
use crate::{FloatNdArrayElement, UnsafeSharedRef, iter_range_par, run_par};

/// Sample a tensor using grid-based sampling.
///
/// # Arguments
///
/// * `tensor` - The tensor being sampled from, shape (N, C, H_in, W_in)
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

    let align_corners = options.align_corners;
    let padding_mode = options.padding_mode;

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
            let (px, py) = if align_corners {
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
                bilinear_interpolate(&tensor, b, c, px, py, width_in, height_in, padding_mode);

            unsafe {
                let output = unsafe_shared_out.get();
                output[(b, c, y, x)] = val.elem();
            }
        });
    });

    output.into_dyn().into_shared()
}

/// Bilinear interpolation at a point with configurable padding mode.
fn bilinear_interpolate<E, S>(
    source: &ndarray::ArrayBase<S, ndarray::Dim<[usize; 4]>>,
    b: usize,
    c: usize,
    x: f64,
    y: f64,
    width: usize,
    height: usize,
    padding_mode: GridSamplePaddingMode,
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
            let x = reflect_coordinate(x, width as f64);
            let y = reflect_coordinate(y, height as f64);
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

/// Reflect a coordinate at the boundaries.
///
/// Uses the formula for reflection padding where coordinates bounce back at boundaries.
fn reflect_coordinate(coord: f64, size: f64) -> f64 {
    if size <= 1.0 {
        return 0.0;
    }

    // Handle reflection for out-of-bounds coordinates
    let mut coord = coord;

    // First handle negative values
    if coord < 0.0 {
        coord = -coord;
    }

    // Then handle values >= size by reflecting
    let max_val = size - 1.0;
    if coord > max_val {
        // Reflect: 2*max - coord, then clamp
        coord = 2.0 * max_val - coord;
        // If still out of bounds after one reflection, clamp
        coord = coord.clamp(0.0, max_val);
    }

    coord
}
