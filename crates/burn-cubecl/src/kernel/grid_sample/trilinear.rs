use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::std::FastDivmod;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{kernel::utils::address_type, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::{Shape, ops::GridSampleOptions};

use super::base::{PaddingMode, fetch_value_3d, reflect_coord};

/// Grid sample with trilinear interpolation.
///
/// Each thread processes all channels for one spatial output position:
/// 1. Reading (x, y, z) coordinates from the grid tensor (once per spatial position)
/// 2. Converting normalized [-1, 1] coords to voxel coordinates (once)
/// 3. For each channel: fetch 8 corner values, interpolate, and write output
///
/// Corner values are named `v_zyx`, where each digit is 0 for the lower index on that axis and
/// 1 for the upper one — `v101` is `(z1, y0, x1)`.
#[cube(launch, address_type = "dynamic")]
fn grid_sample_trilinear_kernel<F: Float>(
    input: &Tensor<F>,                          // [N, C, D_in, H_in, W_in]
    grid: &Tensor<F>,                           // [N, D_out, H_out, W_out, 3]
    output: &mut Tensor<F>,                     // [N, C, D_out, H_out, W_out]
    shape_spatial: Sequence<FastDivmod<usize>>, // [N, D_out, H_out, W_out] for thread decomposition
    #[comptime] align_corners: bool,
    #[comptime] pad_mode: PaddingMode,
    #[define(F)] _dtype: ElemType,
) {
    // Thread index maps to spatial position (n, d_out, h_out, w_out) only
    let spatial_idx = ABSOLUTE_POS;
    let num_spatial = output.shape(0) * output.shape(2) * output.shape(3) * output.shape(4);
    if spatial_idx >= num_spatial {
        terminate!();
    }

    // Decompose spatial index into (n, d_out, h_out, w_out)
    let (rem, w_out) = shape_spatial[3].div_mod(spatial_idx);
    let (rem, h_out) = shape_spatial[2].div_mod(rem);
    let (n, d_out) = shape_spatial[1].div_mod(rem);

    let channels = input.shape(1) as u32;
    let d_in = input.shape(2) as u32;
    let h_in = input.shape(3) as u32;
    let w_in = input.shape(4) as u32;

    // Read grid coordinates once per spatial position.
    //
    // The component axis is strided like every other one rather than assumed
    // contiguous: a caller may legitimately hand in a view such as
    // `[N, 3, D, H, W].permute([0, 2, 3, 4, 1])`, which has the required
    // `[N, D, H, W, 3]` shape but `stride(4) != 1`. Reading `offset + 1` there
    // would pick up the neighbouring *storage* element instead of the y/z
    // coordinate, and diverge from the ndarray backend, which indexes the view
    // logically.
    let grid_offset = n * grid.stride(0)
        + d_out * grid.stride(1)
        + h_out * grid.stride(2)
        + w_out * grid.stride(3);
    let grid_stride_c = grid.stride(4);
    let gx = grid[grid_offset]; // x coordinate in [-1, 1], indexes W_in
    let gy = grid[grid_offset + grid_stride_c]; // y coordinate in [-1, 1], indexes H_in
    let gz = grid[grid_offset + grid_stride_c + grid_stride_c]; // z coordinate in [-1, 1], indexes D_in

    // Convert normalized coordinates to voxel coordinates
    let (px, py, pz) = if align_corners {
        let px = (gx + F::new(1.0_f32)) * F::cast_from((w_in - 1) as f32) / F::new(2.0_f32);
        let py = (gy + F::new(1.0_f32)) * F::cast_from((h_in - 1) as f32) / F::new(2.0_f32);
        let pz = (gz + F::new(1.0_f32)) * F::cast_from((d_in - 1) as f32) / F::new(2.0_f32);
        (px, py, pz)
    } else {
        let px =
            (gx + F::new(1.0_f32)) * F::cast_from(w_in as f32) / F::new(2.0_f32) - F::new(0.5_f32);
        let py =
            (gy + F::new(1.0_f32)) * F::cast_from(h_in as f32) / F::new(2.0_f32) - F::new(0.5_f32);
        let pz =
            (gz + F::new(1.0_f32)) * F::cast_from(d_in as f32) / F::new(2.0_f32) - F::new(0.5_f32);
        (px, py, pz)
    };

    // For reflection padding, reflect the coordinate into the valid sampling range.
    // This ensures integer indices are at most 1 step out of bounds.
    let (px, py, pz) = if comptime!(pad_mode == PaddingMode::Reflection) {
        let px = reflect_coord::<F>(px, w_in, align_corners);
        let py = reflect_coord::<F>(py, h_in, align_corners);
        let pz = reflect_coord::<F>(pz, d_in, align_corners);
        (px, py, pz)
    } else {
        (px, py, pz)
    };

    // Compute floor and ceil indices
    let x0_f = px.floor();
    let y0_f = py.floor();
    let z0_f = pz.floor();
    let x1_f = x0_f + F::new(1.0_f32);
    let y1_f = y0_f + F::new(1.0_f32);
    let z1_f = z0_f + F::new(1.0_f32);

    // Compute interpolation weights
    let wx = px - x0_f;
    let wy = py - y0_f;
    let wz = pz - z0_f;
    let wx_ = F::new(1.0_f32) - wx;
    let wy_ = F::new(1.0_f32) - wy;
    let wz_ = F::new(1.0_f32) - wz;

    // Convert to integers for indexing
    let x0 = i32::cast_from(x0_f);
    let y0 = i32::cast_from(y0_f);
    let z0 = i32::cast_from(z0_f);
    let x1 = i32::cast_from(x1_f);
    let y1 = i32::cast_from(y1_f);
    let z1 = i32::cast_from(z1_f);

    let w_in = w_in as i32;
    let h_in = h_in as i32;
    let d_in = d_in as i32;

    // Pre-compute strides
    let stride_n = input.stride(0);
    let stride_c = input.stride(1);
    let stride_d = input.stride(2);
    let stride_h = input.stride(3);
    let stride_w = input.stride(4);
    let out_stride_n = output.stride(0);
    let out_stride_c = output.stride(1);
    let out_stride_d = output.stride(2);
    let out_stride_h = output.stride(3);
    let out_stride_w = output.stride(4);

    // Base offsets for this spatial position
    let in_base_n = n * stride_n;
    let out_base_spatial =
        n * out_stride_n + d_out * out_stride_d + h_out * out_stride_h + w_out * out_stride_w;

    // Loop over all channels - grid coords and weights are reused
    for c in 0..channels {
        let in_base = in_base_n + c as usize * stride_c;

        let v000 = fetch_value_3d(
            input, in_base, stride_d, stride_h, stride_w, z0, y0, x0, d_in, h_in, w_in, pad_mode,
        );
        let v001 = fetch_value_3d(
            input, in_base, stride_d, stride_h, stride_w, z0, y0, x1, d_in, h_in, w_in, pad_mode,
        );
        let v010 = fetch_value_3d(
            input, in_base, stride_d, stride_h, stride_w, z0, y1, x0, d_in, h_in, w_in, pad_mode,
        );
        let v011 = fetch_value_3d(
            input, in_base, stride_d, stride_h, stride_w, z0, y1, x1, d_in, h_in, w_in, pad_mode,
        );
        let v100 = fetch_value_3d(
            input, in_base, stride_d, stride_h, stride_w, z1, y0, x0, d_in, h_in, w_in, pad_mode,
        );
        let v101 = fetch_value_3d(
            input, in_base, stride_d, stride_h, stride_w, z1, y0, x1, d_in, h_in, w_in, pad_mode,
        );
        let v110 = fetch_value_3d(
            input, in_base, stride_d, stride_h, stride_w, z1, y1, x0, d_in, h_in, w_in, pad_mode,
        );
        let v111 = fetch_value_3d(
            input, in_base, stride_d, stride_h, stride_w, z1, y1, x1, d_in, h_in, w_in, pad_mode,
        );

        // Trilinear interpolation
        let result = wz_ * wy_ * wx_ * v000
            + wz_ * wy_ * wx * v001
            + wz_ * wy * wx_ * v010
            + wz_ * wy * wx * v011
            + wz * wy_ * wx_ * v100
            + wz * wy_ * wx * v101
            + wz * wy * wx_ * v110
            + wz * wy * wx * v111;

        let out_idx = out_base_spatial + c as usize * out_stride_c;
        output[out_idx] = result;
    }
}

/// Launch the grid sample trilinear kernel
pub(crate) fn grid_sample_trilinear_launch(
    input: CubeTensor,
    grid: CubeTensor,
    options: GridSampleOptions,
) -> CubeTensor {
    let [batch_size, channels, _d_in, _h_in, _w_in] = input.meta.shape().dims();
    let [n, d_out, h_out, w_out, three] = grid.meta.shape().dims();
    assert_eq!(three, 3, "Grid last dimension must be 3");
    // The output is allocated and the kernel launched over the *input's* batch,
    // so a shorter grid would be read past its storage and a longer one
    // silently truncated. The ndarray backend rejects the same mismatch.
    assert_eq!(
        batch_size, n,
        "Input batch ({batch_size}) and grid batch ({n}) must match"
    );

    // Create output tensor [N, C, D_out, H_out, W_out]
    let output_shape = Shape::new([batch_size, channels, d_out, h_out, w_out]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );

    // Spatial threading: one thread per (n, d_out, h_out, w_out)
    let spatial_shape = Shape::new([batch_size, d_out, h_out, w_out]);
    let num_spatial = spatial_shape.num_elements();

    let mut shape_spatial = SequenceArg::new();
    for dim in spatial_shape.iter() {
        shape_spatial.push(*dim);
    }

    let cube_dim = CubeDim::new(&input.client, num_spatial);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_spatial, cube_dim);

    let padding_mode: PaddingMode = options.padding_mode.into();

    let dtype = input.dtype;

    grid_sample_trilinear_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(input, grid, output),
        input.into_tensor_arg(),
        grid.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        shape_spatial,
        options.align_corners,
        padding_mode,
        dtype_to_storage_type(dtype),
    );

    output
}
