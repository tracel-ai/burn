use cubecl::std::{FastDivmod, FastDivmodArgs};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    CubeRuntime, kernel::utils::address_type, ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use burn_backend::{Shape, ops::GridSampleOptions};

use super::base::{PaddingMode, fetch_value, reflect_coord};

/// Grid sample with bilinear interpolation.
///
/// Each thread processes all channels for one spatial output position:
/// 1. Reading (x, y) coordinates from the grid tensor (once per spatial position)
/// 2. Converting normalized [-1, 1] coords to pixel coordinates (once)
/// 3. For each channel: fetch 4 corner values, interpolate, and write output
#[cube(launch, address_type = "dynamic")]
fn grid_sample_bilinear_kernel<F: Float>(
    input: &Tensor<F>,                          // [N, C, H_in, W_in]
    grid: &Tensor<F>,                           // [N, H_out, W_out, 2]
    output: &mut Tensor<F>,                     // [N, C, H_out, W_out]
    shape_spatial: Sequence<FastDivmod<usize>>, // [N, H_out, W_out] for thread decomposition
    #[comptime] align_corners: bool,
    #[comptime] pad_mode: PaddingMode,
    #[define(F)] _dtype: StorageType,
) {
    // Thread index maps to spatial position (n, h_out, w_out) only
    let spatial_idx = ABSOLUTE_POS;
    let num_spatial = output.shape(0) * output.shape(2) * output.shape(3);
    if spatial_idx >= num_spatial {
        terminate!();
    }

    // Decompose spatial index into (n, h_out, w_out)
    let (rem, w_out) = shape_spatial[2].div_mod(spatial_idx);
    let (n, h_out) = shape_spatial[1].div_mod(rem);

    let channels = input.shape(1) as u32;
    let h_in = input.shape(2) as u32;
    let w_in = input.shape(3) as u32;

    // Read grid coordinates once per spatial position
    let grid_offset = n * grid.stride(0) + h_out * grid.stride(1) + w_out * grid.stride(2);
    let gx = grid[grid_offset]; // x coordinate in [-1, 1]
    let gy = grid[grid_offset + 1]; // y coordinate in [-1, 1]

    // Convert normalized coordinates to pixel coordinates
    let (px, py) = if align_corners {
        let px = (gx + F::new(1.0)) * F::cast_from((w_in - 1) as f32) / F::new(2.0);
        let py = (gy + F::new(1.0)) * F::cast_from((h_in - 1) as f32) / F::new(2.0);
        (px, py)
    } else {
        let px = (gx + F::new(1.0)) * F::cast_from(w_in as f32) / F::new(2.0) - F::new(0.5);
        let py = (gy + F::new(1.0)) * F::cast_from(h_in as f32) / F::new(2.0) - F::new(0.5);
        (px, py)
    };

    // For reflection padding, reflect the coordinate into the valid sampling range.
    // This ensures integer indices are at most 1 step out of bounds.
    let (px, py) = if comptime!(pad_mode == PaddingMode::Reflection) {
        let px = reflect_coord::<F>(px, w_in, align_corners);
        let py = reflect_coord::<F>(py, h_in, align_corners);
        (px, py)
    } else {
        (px, py)
    };

    // Compute floor and ceil indices
    let x0_f = px.floor();
    let y0_f = py.floor();
    let x1_f = x0_f + F::new(1.0);
    let y1_f = y0_f + F::new(1.0);

    // Compute interpolation weights
    let wx = px - x0_f;
    let wy = py - y0_f;
    let wx_ = F::new(1.0) - wx;
    let wy_ = F::new(1.0) - wy;

    // Convert to integers for indexing
    let x0 = i32::cast_from(x0_f);
    let y0 = i32::cast_from(y0_f);
    let x1 = i32::cast_from(x1_f);
    let y1 = i32::cast_from(y1_f);

    let w_in = w_in as i32;
    let h_in = h_in as i32;

    // Pre-compute strides
    let stride_n = input.stride(0);
    let stride_c = input.stride(1);
    let stride_h = input.stride(2);
    let stride_w = input.stride(3);
    let out_stride_n = output.stride(0);
    let out_stride_c = output.stride(1);
    let out_stride_h = output.stride(2);
    let out_stride_w = output.stride(3);

    // Base offsets for this spatial position
    let in_base_n = n * stride_n;
    let out_base_spatial = n * out_stride_n + h_out * out_stride_h + w_out * out_stride_w;

    // Loop over all channels - grid coords and weights are reused
    for c in 0..channels {
        let in_base = in_base_n + c as usize * stride_c;

        let v00 = fetch_value(
            input, in_base, stride_h, stride_w, y0, x0, h_in, w_in, pad_mode,
        );
        let v01 = fetch_value(
            input, in_base, stride_h, stride_w, y1, x0, h_in, w_in, pad_mode,
        );
        let v10 = fetch_value(
            input, in_base, stride_h, stride_w, y0, x1, h_in, w_in, pad_mode,
        );
        let v11 = fetch_value(
            input, in_base, stride_h, stride_w, y1, x1, h_in, w_in, pad_mode,
        );

        // Bilinear interpolation
        let result = wx_ * wy_ * v00 + wx_ * wy * v01 + wx * wy_ * v10 + wx * wy * v11;

        let out_idx = out_base_spatial + c as usize * out_stride_c;
        output[out_idx] = result;
    }
}

/// Launch the grid sample bilinear kernel
pub(crate) fn grid_sample_bilinear_launch<R: CubeRuntime>(
    input: CubeTensor<R>,
    grid: CubeTensor<R>,
    options: GridSampleOptions,
) -> CubeTensor<R> {
    let [batch_size, channels, _h_in, _w_in] = input.meta.shape().dims();
    let [_n, h_out, w_out, two] = grid.meta.shape().dims();
    assert_eq!(two, 2, "Grid last dimension must be 2");

    // Create output tensor [N, C, H_out, W_out]
    let output_shape = Shape::new([batch_size, channels, h_out, w_out]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );

    // Spatial threading: one thread per (n, h_out, w_out)
    let spatial_shape = Shape::new([batch_size, h_out, w_out]);
    let num_spatial = spatial_shape.num_elements();

    let mut shape_spatial = SequenceArg::new();
    for dim in spatial_shape.iter() {
        shape_spatial.push(FastDivmodArgs::new(&input.client, *dim));
    }

    let cube_dim = CubeDim::new(&input.client, num_spatial);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_spatial, cube_dim);

    let padding_mode: PaddingMode = options.padding_mode.into();

    let dtype = input.dtype;

    grid_sample_bilinear_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(input, grid, output),
        input.into_tensor_arg(1),
        grid.into_tensor_arg(1),
        output.clone().into_tensor_arg(1),
        shape_spatial,
        options.align_corners,
        padding_mode,
        dtype.into(),
    );

    output
}
