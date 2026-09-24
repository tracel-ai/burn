use crate::{
    Backend, TensorMetadata, get_device_settings,
    ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode},
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use alloc::vec;
use alloc::vec::Vec;
use burn_std::{BoolDType, FloatDType, IndexingUpdateOp, IntDType, Shape, Slice};

/// Reference implementation of grid_sample_2d that supports all options.
///
/// # Arguments
///
/// * `tensor` - The tensor being sampled from, must be contiguous with shape (N, C, H_in, W_in)
/// * `grid` - A tensor of locations, with shape (N, H_out, W_out, 2). Values are [-1, 1].
///   A [x = -1, y = -1] means top-left, and [x = 1, y = 1] means bottom-right
/// * `options` - Grid sampling options
///
/// # Returns
///
/// A tensor with shape (N, C, H_out, W_out)
pub fn float_grid_sample_2d_ref<B: Backend>(
    tensor: FloatTensor<B>,
    grid: FloatTensor<B>,
    options: GridSampleOptions,
) -> FloatTensor<B> {
    match options.mode {
        InterpolateMode::Bilinear => float_grid_sample_2d_bilinear::<B>(
            tensor,
            grid,
            options.padding_mode,
            options.align_corners,
        ),
        _ => todo!(
            "Default implementation for grid_sample_2d with {:?} unimplemented",
            options.mode
        ),
    }
}

/// Bilinear grid sampling implementation.
fn float_grid_sample_2d_bilinear<B: Backend>(
    tensor: FloatTensor<B>,
    grid: FloatTensor<B>,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> FloatTensor<B> {
    let n = tensor.shape()[0];
    let c = tensor.shape()[1];
    let h_in = tensor.shape()[2];
    let w_in = tensor.shape()[3];
    let h_out = grid.shape()[1];
    let w_out = grid.shape()[2];
    let spatial_in = h_in * w_in;
    let spatial_out = h_out * w_out;
    let device = tensor.device();

    // Separate x and y coordinates from grid
    // shape: (N, H_out, W_out, 1)
    let grid_x_slice = vec![
        Slice::new(0, Some(n as isize), 1),
        Slice::new(0, Some(h_out as isize), 1),
        Slice::new(0, Some(w_out as isize), 1),
        Slice::new(0, Some(1), 1),
    ];
    let grid_y_slice = vec![
        Slice::new(0, Some(n as isize), 1),
        Slice::new(0, Some(h_out as isize), 1),
        Slice::new(0, Some(w_out as isize), 1),
        Slice::new(1, Some(2), 1),
    ];

    let grid_x = B::float_slice(grid.clone(), &grid_x_slice);
    let grid_x = B::float_reshape(grid_x, Shape::new([n, 1, h_out, w_out]));
    let grid_y = B::float_slice(grid.clone(), &grid_y_slice);
    let grid_y = B::float_reshape(grid_y, Shape::new([n, 1, h_out, w_out]));

    // Convert normalized grid coordinates [-1, 1] to pixel coordinates
    let w_in_f = w_in as f64;
    let h_in_f = h_in as f64;

    let (grid_x, grid_y) = if align_corners {
        // align_corners=true: x_pixel = (x_norm + 1) * (width - 1) / 2
        // Maps -1 to 0 and 1 to width - 1
        let grid_x = B::float_add_scalar(grid_x, 1f32.into());
        let grid_x = B::float_mul_scalar(grid_x, ((w_in_f - 1.0) / 2.0).into());

        let grid_y = B::float_add_scalar(grid_y, 1f32.into());
        let grid_y = B::float_mul_scalar(grid_y, ((h_in_f - 1.0) / 2.0).into());

        (grid_x, grid_y)
    } else {
        // align_corners=false: x_pixel = (x_norm + 1) * width / 2 - 0.5
        // Maps -1 to -0.5 and 1 to width - 0.5
        let grid_x = B::float_add_scalar(grid_x, 1f32.into());
        let grid_x = B::float_mul_scalar(grid_x, (w_in_f / 2.0).into());
        let grid_x = B::float_sub_scalar(grid_x, 0.5f32.into());

        let grid_y = B::float_add_scalar(grid_y, 1f32.into());
        let grid_y = B::float_mul_scalar(grid_y, (h_in_f / 2.0).into());
        let grid_y = B::float_sub_scalar(grid_y, 0.5f32.into());

        (grid_x, grid_y)
    };

    // Apply padding mode to coordinates
    let (grid_x, grid_y) = match padding_mode {
        GridSamplePaddingMode::Border => {
            // Clamp coordinates to valid range [0, size-1]
            let grid_x = B::float_clamp(grid_x, 0f32.into(), ((w_in - 1) as f32).into());
            let grid_y = B::float_clamp(grid_y, 0f32.into(), ((h_in - 1) as f32).into());
            (grid_x, grid_y)
        }
        GridSamplePaddingMode::Reflection => {
            // Reflect coordinates at boundaries
            let grid_x = reflect_coordinates::<B>(grid_x, w_in_f, align_corners);
            let grid_y = reflect_coordinates::<B>(grid_y, h_in_f, align_corners);
            (grid_x, grid_y)
        }
        GridSamplePaddingMode::Zeros => {
            // Keep coordinates as-is, we'll mask out-of-bounds later
            (grid_x, grid_y)
        }
    };

    // Get floor indices for the four corners
    let grid_x_floored = B::float_floor(grid_x.clone());
    let grid_y_floored = B::float_floor(grid_y.clone());

    // Compute interpolation weights (fractional part)
    let x_frac = B::float_sub(grid_x.clone(), grid_x_floored.clone());
    let y_frac = B::float_sub(grid_y.clone(), grid_y_floored.clone());

    // Convert to integer indices
    let settings = get_device_settings::<B>(&device);
    let x0 = B::float_into_int(grid_x_floored.clone(), settings.int_dtype);
    let y0 = B::float_into_int(grid_y_floored.clone(), settings.int_dtype);
    let x1 = B::float_into_int(
        B::float_add_scalar(grid_x_floored, 1f32.into()),
        settings.int_dtype,
    );
    let y1 = B::float_into_int(
        B::float_add_scalar(grid_y_floored, 1f32.into()),
        settings.int_dtype,
    );

    // Create masks for out-of-bounds coordinates (only used for zeros padding)
    let (mask_00, mask_01, mask_10, mask_11) = if padding_mode == GridSamplePaddingMode::Zeros {
        let x0_valid = B::int_greater_equal_elem(x0.clone(), 0.into(), settings.bool_dtype);
        let x0_valid = B::bool_and(
            x0_valid,
            B::int_lower_elem(x0.clone(), (w_in as i32).into(), settings.bool_dtype),
        );
        let x1_valid = B::int_greater_equal_elem(x1.clone(), 0.into(), settings.bool_dtype);
        let x1_valid = B::bool_and(
            x1_valid,
            B::int_lower_elem(x1.clone(), (w_in as i32).into(), settings.bool_dtype),
        );
        let y0_valid = B::int_greater_equal_elem(y0.clone(), 0.into(), settings.bool_dtype);
        let y0_valid = B::bool_and(
            y0_valid,
            B::int_lower_elem(y0.clone(), (h_in as i32).into(), settings.bool_dtype),
        );
        let y1_valid = B::int_greater_equal_elem(y1.clone(), 0.into(), settings.bool_dtype);
        let y1_valid = B::bool_and(
            y1_valid,
            B::int_lower_elem(y1.clone(), (h_in as i32).into(), settings.bool_dtype),
        );

        (
            Some(B::bool_and(x0_valid.clone(), y0_valid.clone())),
            Some(B::bool_and(x0_valid.clone(), y1_valid.clone())),
            Some(B::bool_and(x1_valid.clone(), y0_valid)),
            Some(B::bool_and(x1_valid, y1_valid)),
        )
    } else {
        (None, None, None, None)
    };

    // Clamp indices to valid range for gather
    let x0_clamped = B::int_clamp(x0, 0.into(), ((w_in - 1) as i32).into());
    let x1_clamped = B::int_clamp(x1, 0.into(), ((w_in - 1) as i32).into());
    let y0_clamped = B::int_clamp(y0, 0.into(), ((h_in - 1) as i32).into());
    let y1_clamped = B::int_clamp(y1, 0.into(), ((h_in - 1) as i32).into());

    // Linear indices: idx = y * W_in + x
    let w_in_scalar: i32 = w_in as i32;
    let idx_00 = B::int_add(
        B::int_mul_scalar(y0_clamped.clone(), w_in_scalar.into()),
        x0_clamped.clone(),
    );
    let idx_01 = B::int_add(
        B::int_mul_scalar(y1_clamped.clone(), w_in_scalar.into()),
        x0_clamped,
    );
    let idx_10 = B::int_add(
        B::int_mul_scalar(y0_clamped, w_in_scalar.into()),
        x1_clamped.clone(),
    );
    let idx_11 = B::int_add(
        B::int_mul_scalar(y1_clamped, w_in_scalar.into()),
        x1_clamped,
    );

    // [N, 1, H_out, W_out] -> [N, 1, H_out * W_out]
    let idx_00 = B::int_reshape(idx_00, Shape::new([n, 1, spatial_out]));
    let idx_01 = B::int_reshape(idx_01, Shape::new([n, 1, spatial_out]));
    let idx_10 = B::int_reshape(idx_10, Shape::new([n, 1, spatial_out]));
    let idx_11 = B::int_reshape(idx_11, Shape::new([n, 1, spatial_out]));

    // [N, 1, spatial] -> [N, C, spatial]
    let idx_00 = B::int_expand(idx_00, Shape::new([n, c, spatial_out]));
    let idx_01 = B::int_expand(idx_01, Shape::new([n, c, spatial_out]));
    let idx_10 = B::int_expand(idx_10, Shape::new([n, c, spatial_out]));
    let idx_11 = B::int_expand(idx_11, Shape::new([n, c, spatial_out]));

    let tensor_flat = B::float_reshape(tensor, Shape::new([n, c, spatial_in]));

    let sample_00 = B::float_gather(2, tensor_flat.clone(), idx_00);
    let sample_01 = B::float_gather(2, tensor_flat.clone(), idx_01);
    let sample_10 = B::float_gather(2, tensor_flat.clone(), idx_10);
    let sample_11 = B::float_gather(2, tensor_flat, idx_11);

    // Reshape samples to (N, C, H_out, W_out)
    let sample_00 = B::float_reshape(sample_00, Shape::new([n, c, h_out, w_out]));
    let sample_01 = B::float_reshape(sample_01, Shape::new([n, c, h_out, w_out]));
    let sample_10 = B::float_reshape(sample_10, Shape::new([n, c, h_out, w_out]));
    let sample_11 = B::float_reshape(sample_11, Shape::new([n, c, h_out, w_out]));

    // Apply masks for zeros padding (set out-of-bounds samples to 0)
    let (sample_00, sample_01, sample_10, sample_11) =
        if padding_mode == GridSamplePaddingMode::Zeros {
            let mask_00 = mask_00.unwrap();
            let mask_01 = mask_01.unwrap();
            let mask_10 = mask_10.unwrap();
            let mask_11 = mask_11.unwrap();

            let mask_00_inv = B::bool_not(mask_00);
            let mask_00_inv = B::bool_reshape(mask_00_inv, Shape::new([n, 1, h_out, w_out]));
            let mask_00_inv = B::bool_expand(mask_00_inv, Shape::new([n, c, h_out, w_out]));
            let mask_01_inv = B::bool_not(mask_01);
            let mask_01_inv = B::bool_reshape(mask_01_inv, Shape::new([n, 1, h_out, w_out]));
            let mask_01_inv = B::bool_expand(mask_01_inv, Shape::new([n, c, h_out, w_out]));
            let mask_10_inv = B::bool_not(mask_10);
            let mask_10_inv = B::bool_reshape(mask_10_inv, Shape::new([n, 1, h_out, w_out]));
            let mask_10_inv = B::bool_expand(mask_10_inv, Shape::new([n, c, h_out, w_out]));
            let mask_11_inv = B::bool_not(mask_11);
            let mask_11_inv = B::bool_reshape(mask_11_inv, Shape::new([n, 1, h_out, w_out]));
            let mask_11_inv = B::bool_expand(mask_11_inv, Shape::new([n, c, h_out, w_out]));

            (
                B::float_mask_fill(sample_00, mask_00_inv, 0f32.into()),
                B::float_mask_fill(sample_01, mask_01_inv, 0f32.into()),
                B::float_mask_fill(sample_10, mask_10_inv, 0f32.into()),
                B::float_mask_fill(sample_11, mask_11_inv, 0f32.into()),
            )
        } else {
            (sample_00, sample_01, sample_10, sample_11)
        };

    // Compute bilinear interpolation weights
    let one_minus_x = B::float_neg(x_frac.clone());
    let one_minus_x = B::float_add_scalar(one_minus_x, 1f32.into());

    let one_minus_y = B::float_neg(y_frac.clone());
    let one_minus_y = B::float_add_scalar(one_minus_y, 1f32.into());

    let weight_00 = B::float_mul(one_minus_x.clone(), one_minus_y.clone());
    let weight_01 = B::float_mul(one_minus_x.clone(), y_frac.clone());
    let weight_10 = B::float_mul(x_frac.clone(), one_minus_y);
    let weight_11 = B::float_mul(x_frac, y_frac);

    // Bilinear interpolation
    let result = B::float_mul(sample_00, weight_00);
    let result = B::float_add(result, B::float_mul(sample_01, weight_01));
    let result = B::float_add(result, B::float_mul(sample_10, weight_10));

    B::float_add(result, B::float_mul(sample_11, weight_11))
}

/// Reflect coordinates at boundaries using a triangle wave pattern.
///
/// For align_corners=true: reflects within [0, size-1]
/// For align_corners=false: reflects within [-0.5, size-0.5]
fn reflect_coordinates<B: Backend>(
    coords: FloatTensor<B>,
    size: f64,
    align_corners: bool,
) -> FloatTensor<B> {
    let (min_val, max_val) = if align_corners {
        (0.0f32, (size - 1.0) as f32)
    } else {
        (-0.5f32, (size - 0.5) as f32)
    };

    let span = max_val - min_val;
    if span <= 0.0 {
        // Edge case: size is 1, just return min_val everywhere
        let zeros = B::float_mul_scalar(coords, 0f32.into());
        return B::float_add_scalar(zeros, min_val.into());
    }

    // Triangle wave formula: span - |((x mod 2*span) - span)| + min_val
    let period = 2.0 * span;

    // x = abs(coord - min_val)
    let x = B::float_sub_scalar(coords, min_val.into());
    let x = B::float_abs(x);

    // x_mod = x - floor(x / period) * period
    let x_div = B::float_div_scalar(x.clone(), period.into());
    let x_div_floor = B::float_floor(x_div);
    let x_mod = B::float_sub(x, B::float_mul_scalar(x_div_floor, period.into()));

    // result = span - abs(x_mod - span) + min_val
    let diff = B::float_sub_scalar(x_mod, span.into());
    let abs_diff = B::float_abs(diff);
    let reflected = B::float_sub_scalar(abs_diff, span.into());
    let reflected = B::float_neg(reflected);
    B::float_add_scalar(reflected, min_val.into())
}

/// Reference implementation of grid_sample_3d that supports all options.
///
/// # Arguments
///
/// * `tensor` - The tensor being sampled from, must be contiguous with shape
///   (N, C, D_in, H_in, W_in)
/// * `grid` - A tensor of locations, with shape (N, D_out, H_out, W_out, 3). Values are [-1, 1]
///   and the last dimension is ordered `(x, y, z)`, where `x` indexes `W_in`, `y` indexes `H_in`
///   and `z` indexes `D_in`. A [x = -1, y = -1, z = -1] means the front-top-left corner, and
///   [x = 1, y = 1, z = 1] the back-bottom-right one
/// * `options` - Grid sampling options
///
/// # Returns
///
/// A tensor with shape (N, C, D_out, H_out, W_out)
pub fn float_grid_sample_3d_ref<B: Backend>(
    tensor: FloatTensor<B>,
    grid: FloatTensor<B>,
    options: GridSampleOptions,
) -> FloatTensor<B> {
    let tensor_shape = tensor.shape();
    let grid_shape = grid.shape();
    assert_eq!(
        tensor_shape.num_dims(),
        5,
        "grid_sample_3d: input must have shape (N, C, D_in, H_in, W_in)"
    );
    assert_eq!(
        grid_shape.num_dims(),
        5,
        "grid_sample_3d: grid must have shape (N, D_out, H_out, W_out, 3)"
    );
    assert_eq!(
        grid_shape[4], 3,
        "grid_sample_3d: grid last dimension must be 3"
    );
    assert_eq!(
        tensor_shape[0], grid_shape[0],
        "grid_sample_3d: input and grid batch sizes must match"
    );

    match options.mode {
        // `InterpolateMode` is shared across ranks, so `Bilinear` selects linear interpolation
        // at whatever rank the op works on; at rank 5 that is trilinear.
        InterpolateMode::Bilinear => float_grid_sample_3d_trilinear::<B>(
            tensor,
            grid,
            options.padding_mode,
            options.align_corners,
        ),
        InterpolateMode::Nearest => float_grid_sample_3d_nearest::<B>(
            tensor,
            grid,
            options.padding_mode,
            options.align_corners,
        ),
        _ => todo!(
            "Default implementation for grid_sample_3d with {:?} unimplemented",
            options.mode
        ),
    }
}

/// Trilinear grid sampling implementation.
fn float_grid_sample_3d_trilinear<B: Backend>(
    tensor: FloatTensor<B>,
    grid: FloatTensor<B>,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> FloatTensor<B> {
    let n = tensor.shape()[0];
    let c = tensor.shape()[1];
    let d_in = tensor.shape()[2];
    let h_in = tensor.shape()[3];
    let w_in = tensor.shape()[4];
    let d_out = grid.shape()[1];
    let h_out = grid.shape()[2];
    let w_out = grid.shape()[3];
    let spatial_in = d_in * h_in * w_in;
    let spatial_out = d_out * h_out * w_out;

    let corners =
        TrilinearCorners3d::<B>::new(grid, [d_in, h_in, w_in], padding_mode, align_corners);

    let tensor_flat = B::float_reshape(tensor, Shape::new([n, c, spatial_in]));
    let out_shape = Shape::new([n, c, d_out, h_out, w_out]);

    let mut result: Option<FloatTensor<B>> = None;

    for corner in CORNERS_3D {
        let sample = gather_voxels::<B>(
            tensor_flat.clone(),
            corners.corner_index(corner),
            [h_in, w_in],
            [n, c, spatial_out],
            out_shape.clone(),
        );

        // Apply masks for zeros padding (set out-of-bounds samples to 0)
        let sample = corners.mask_outside(sample, corner, &out_shape);
        let weighted = B::float_mul(sample, corners.corner_weight(corner));

        result = Some(match result {
            Some(acc) => B::float_add(acc, weighted),
            None => weighted,
        });
    }

    result.expect("Trilinear interpolation always accumulates the eight corners")
}

/// Nearest-neighbor grid sampling implementation.
///
/// Ties (coordinates landing exactly on `.5`) follow `float_round`, which rounds half to even and
/// so matches PyTorch's `std::nearbyint`. Half-integer coordinates are the *common* case when
/// `align_corners` is false, so backends overriding this path must break ties the same way.
fn float_grid_sample_3d_nearest<B: Backend>(
    tensor: FloatTensor<B>,
    grid: FloatTensor<B>,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> FloatTensor<B> {
    let n = tensor.shape()[0];
    let c = tensor.shape()[1];
    let d_in = tensor.shape()[2];
    let h_in = tensor.shape()[3];
    let w_in = tensor.shape()[4];
    let d_out = grid.shape()[1];
    let h_out = grid.shape()[2];
    let w_out = grid.shape()[3];
    let spatial_in = d_in * h_in * w_in;
    let spatial_out = d_out * h_out * w_out;

    let (indices, outside) =
        nearest_voxel_3d::<B>(grid, [d_in, h_in, w_in], padding_mode, align_corners);

    let out_shape = Shape::new([n, c, d_out, h_out, w_out]);
    let tensor_flat = B::float_reshape(tensor, Shape::new([n, c, spatial_in]));
    let sample = gather_voxels::<B>(
        tensor_flat,
        indices,
        [h_in, w_in],
        [n, c, spatial_out],
        out_shape.clone(),
    );

    match outside {
        Some(mask) => B::float_mask_fill(sample, B::bool_expand(mask, out_shape), 0f32.into()),
        None => sample,
    }
}

/// Gradient of [`float_grid_sample_3d_ref`] with respect to the sampled tensor.
///
/// # Arguments
///
/// * `x_shape` - Shape of the sampled tensor, (N, C, D_in, H_in, W_in)
/// * `grid` - The same sampling grid the forward was given, (N, D_out, H_out, W_out, 3)
/// * `grad_output` - Gradient of the loss w.r.t. the output, (N, C, D_out, H_out, W_out)
/// * `options` - The same options the forward was given
///
/// # Returns
///
/// A tensor with shape `x_shape`.
///
/// # Derivation
///
/// The forward reads
///
/// ```text
/// out[n, c, o] = sum over the 8 corners of  mu * V[n, c, corner] * w_x * w_y * w_z
/// ```
///
/// where `mu` is 1 unless `Zeros` padding put the corner outside the volume. Only `V` depends on
/// the sampled tensor, so the transpose of that sum scatter-adds `grad_output * mu * w_x w_y w_z`
/// back onto each corner — the same eight index sets the forward gathered from.
///
/// The masking is load-bearing here, not just in the forward: an out-of-bounds corner index is
/// *clamped* onto a real voxel before the gather, so without zeroing its weight the scatter would
/// deposit gradient on a voxel the forward never read.
pub fn float_grid_sample_3d_x_backward<B: Backend>(
    x_shape: Shape,
    grid: FloatTensor<B>,
    grad_output: FloatTensor<B>,
    options: GridSampleOptions,
) -> FloatTensor<B> {
    match options.mode {
        InterpolateMode::Bilinear => grid_sample_3d_trilinear_x_backward::<B>(
            x_shape,
            grid,
            grad_output,
            options.padding_mode,
            options.align_corners,
        ),
        InterpolateMode::Nearest => grid_sample_3d_nearest_x_backward::<B>(
            x_shape,
            grid,
            grad_output,
            options.padding_mode,
            options.align_corners,
        ),
        _ => todo!(
            "Backward pass for grid_sample_3d with {:?} unimplemented",
            options.mode
        ),
    }
}

/// Gradient of [`float_grid_sample_3d_ref`] with respect to the sampling grid.
///
/// # Arguments
///
/// * `tensor` - The same sampled tensor the forward was given, (N, C, D_in, H_in, W_in)
/// * `grid` - The same sampling grid the forward was given, (N, D_out, H_out, W_out, 3)
/// * `grad_output` - Gradient of the loss w.r.t. the output, (N, C, D_out, H_out, W_out)
/// * `options` - The same options the forward was given
///
/// # Returns
///
/// A tensor with the same shape as `grid`, with the last dimension ordered `(x, y, z)`.
///
/// # Derivation
///
/// Of the forward's factors only the interpolation weights depend on the grid, and they are
/// separable and piecewise linear: with `w_0 = 1 - t` and `w_1 = t`, the derivative of a corner's
/// weight along one axis is `-1` for the lower neighbour and `+1` for the upper one, leaving the
/// other two axes' weights untouched. So
///
/// ```text
/// d out / d q_x = sum over corners  mu * V[corner] * (corner picked x_lo ? -1 : +1) * w_y * w_z
/// ```
///
/// The same grid coordinate drives every channel of an output position, so the channel axis is
/// summed. Chaining back to the raw grid value multiplies by `d q / d p` (the padding-mode
/// derivative) and then by `d p / d g` — the `[-1, 1]` to pixel scale
/// `(size - 1) / 2` when `align_corners` and `size / 2` otherwise.
///
/// # PyTorch parity
///
/// The weight derivatives, the channel sum, the `[-1, 1]` scaling, `Border`'s gradient-free
/// borders and `Reflection`'s sign flips — including both tie conventions, at `p == min` and at
/// `x_mod == span` — all match `grid_sampler_3d_backward`.
///
/// Reflection reaches that agreement by a slightly different route. PyTorch clips the reflected
/// coordinate to `[0, size - 1]` afterwards and zeroes the gradient wherever that clip bites,
/// while this crate's `reflect_coordinates` does not clip: it leaves the coordinate in
/// `[-0.5, size - 0.5]` and lets the corner indices clamp. Strictly outside `[0, size - 1]` the
/// two coincide on their own, because both corners clamp onto the *same* voxel with weights
/// summing to one, so the `-1` and `+1` weight derivatives cancel. Exactly *on* a border they do
/// not — the corners are two distinct voxels — so [`pad_coordinate_grad`] applies PyTorch's
/// clip rule there and zeroes the gradient, the same rule `Border` uses.
pub fn float_grid_sample_3d_grid_backward<B: Backend>(
    tensor: FloatTensor<B>,
    grid: FloatTensor<B>,
    grad_output: FloatTensor<B>,
    options: GridSampleOptions,
) -> FloatTensor<B> {
    match options.mode {
        InterpolateMode::Bilinear => grid_sample_3d_trilinear_grid_backward::<B>(
            tensor,
            grid,
            grad_output,
            options.padding_mode,
            options.align_corners,
        ),
        // Nearest sampling is piecewise constant in the grid — the forward runs the coordinates
        // through `float_round` and `float_into_int`, both of which have a zero derivative almost
        // everywhere. PyTorch returns zeros here too.
        InterpolateMode::Nearest => {
            let device = grid.device();
            let dtype = grid.dtype().into();
            B::float_zeros(grid.shape(), &device, dtype)
        }
        _ => todo!(
            "Backward pass for grid_sample_3d with {:?} unimplemented",
            options.mode
        ),
    }
}

/// Input gradient of the trilinear forward: eight weighted scatter-adds, one per corner.
fn grid_sample_3d_trilinear_x_backward<B: Backend>(
    x_shape: Shape,
    grid: FloatTensor<B>,
    grad_output: FloatTensor<B>,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> FloatTensor<B> {
    let n = x_shape[0];
    let c = x_shape[1];
    let d_in = x_shape[2];
    let h_in = x_shape[3];
    let w_in = x_shape[4];
    let d_out = grid.shape()[1];
    let h_out = grid.shape()[2];
    let w_out = grid.shape()[3];
    let spatial_in = d_in * h_in * w_in;
    let spatial_out = d_out * h_out * w_out;
    let device = grad_output.device();
    let dtype = grad_output.dtype().into();

    let corners =
        TrilinearCorners3d::<B>::new(grid, [d_in, h_in, w_in], padding_mode, align_corners);

    let out_shape = Shape::new([n, c, d_out, h_out, w_out]);
    let flat_out = Shape::new([n, c, spatial_out]);
    let mut grad_x = B::float_zeros(Shape::new([n, c, spatial_in]), &device, dtype);

    for corner in CORNERS_3D {
        let contribution = B::float_mul(grad_output.clone(), corners.corner_weight(corner));
        let contribution = corners.mask_outside(contribution, corner, &out_shape);
        let contribution = B::float_reshape(contribution, flat_out.clone());

        let indices = voxel_indices::<B>(
            corners.corner_index(corner),
            [h_in, w_in],
            [n, c, spatial_out],
        );

        grad_x = B::float_scatter(2, grad_x, indices, contribution, IndexingUpdateOp::Add);
    }

    B::float_reshape(grad_x, x_shape)
}

/// Input gradient of the nearest forward: a single scatter-add at the rounded voxel.
fn grid_sample_3d_nearest_x_backward<B: Backend>(
    x_shape: Shape,
    grid: FloatTensor<B>,
    grad_output: FloatTensor<B>,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> FloatTensor<B> {
    let n = x_shape[0];
    let c = x_shape[1];
    let d_in = x_shape[2];
    let h_in = x_shape[3];
    let w_in = x_shape[4];
    let d_out = grid.shape()[1];
    let h_out = grid.shape()[2];
    let w_out = grid.shape()[3];
    let spatial_in = d_in * h_in * w_in;
    let spatial_out = d_out * h_out * w_out;
    let device = grad_output.device();
    let dtype = grad_output.dtype().into();

    let (indices, outside) =
        nearest_voxel_3d::<B>(grid, [d_in, h_in, w_in], padding_mode, align_corners);

    let out_shape = Shape::new([n, c, d_out, h_out, w_out]);
    let grad_output = match outside {
        Some(mask) => B::float_mask_fill(grad_output, B::bool_expand(mask, out_shape), 0f32.into()),
        None => grad_output,
    };
    let grad_output = B::float_reshape(grad_output, Shape::new([n, c, spatial_out]));

    let indices = voxel_indices::<B>(indices, [h_in, w_in], [n, c, spatial_out]);
    let grad_x = B::float_zeros(Shape::new([n, c, spatial_in]), &device, dtype);
    let grad_x = B::float_scatter(2, grad_x, indices, grad_output, IndexingUpdateOp::Add);

    B::float_reshape(grad_x, x_shape)
}

/// Grid gradient of the trilinear forward.
fn grid_sample_3d_trilinear_grid_backward<B: Backend>(
    tensor: FloatTensor<B>,
    grid: FloatTensor<B>,
    grad_output: FloatTensor<B>,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> FloatTensor<B> {
    let n = tensor.shape()[0];
    let c = tensor.shape()[1];
    let d_in = tensor.shape()[2];
    let h_in = tensor.shape()[3];
    let w_in = tensor.shape()[4];
    let d_out = grid.shape()[1];
    let h_out = grid.shape()[2];
    let w_out = grid.shape()[3];
    let spatial_in = d_in * h_in * w_in;
    let spatial_out = d_out * h_out * w_out;
    let extent = [w_in as f64, h_in as f64, d_in as f64];

    // The coordinates *before* padding: the padding derivative is a function of those.
    let raw = grid_sample_3d_raw_coords::<B>(grid.clone(), [d_in, h_in, w_in], align_corners);
    let corners =
        TrilinearCorners3d::<B>::new(grid, [d_in, h_in, w_in], padding_mode, align_corners);

    let tensor_flat = B::float_reshape(tensor, Shape::new([n, c, spatial_in]));
    let out_shape = Shape::new([n, c, d_out, h_out, w_out]);

    // d(output) / d(padded coordinate), per axis, accumulated over the eight corners.
    let mut d_coord: [Option<FloatTensor<B>>; 3] = [None, None, None];

    for corner in CORNERS_3D {
        let sample = gather_voxels::<B>(
            tensor_flat.clone(),
            corners.corner_index(corner),
            [h_in, w_in],
            [n, c, spatial_out],
            out_shape.clone(),
        );
        let sample = corners.mask_outside(sample, corner, &out_shape);

        // One grid coordinate drives every channel of the same output position, so the channel
        // axis is summed once per corner rather than once per axis.
        let weighted = B::float_sum_dim(B::float_mul(grad_output.clone(), sample), 1);

        for (axis, accumulator) in d_coord.iter_mut().enumerate() {
            let term = B::float_mul(weighted.clone(), corners.cross_weight(corner, axis));
            // d w / d q is -1 for the lower neighbour along this axis and +1 for the upper one.
            let term = if corner[axis] == 0 {
                B::float_neg(term)
            } else {
                term
            };

            *accumulator = Some(match accumulator.take() {
                Some(sum) => B::float_add(sum, term),
                None => term,
            });
        }
    }

    // (N, 1, D_out, H_out, W_out) per axis -> (N, D_out, H_out, W_out, 1), concatenated as
    // `(x, y, z)` on the last axis to rebuild the grid's layout.
    let coord_shape = Shape::new([n, d_out, h_out, w_out, 1]);
    let grads = d_coord
        .into_iter()
        .zip(raw)
        .zip(extent)
        .map(|((accumulator, raw), size)| {
            let grad = accumulator.expect("Every axis accumulates all eight corners");
            let grad = match pad_coordinate_grad::<B>(raw, size, padding_mode, align_corners) {
                Some(d_pad) => B::float_mul(grad, d_pad),
                None => grad,
            };
            let grad = B::float_mul_scalar(grad, coord_scale(size, align_corners).into());

            B::float_reshape(grad, coord_shape.clone())
        })
        .collect::<Vec<_>>();

    B::float_cat(grads, 4)
}

/// The eight corners of the voxel bracketing a sample position, each written as the per-axis
/// `[x, y, z]` choice of the lower (`0`) or upper (`1`) neighbour.
///
/// The order is `x` fastest and `z` slowest, so accumulating over it reproduces the
/// `for z { for y { for x } }` summation order the forward has always used.
const CORNERS_3D: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// The per-axis corner indices and interpolation weights that both the trilinear forward and its
/// backward are built from.
///
/// Every field is indexed `[axis][corner]`, with axis `0 = x`, `1 = y`, `2 = z` and corner `0` the
/// lower (floored) neighbour, `1` the upper one. Each tensor has shape
/// (N, 1, D_out, H_out, W_out); the singleton channel axis lets it broadcast against `C`.
struct TrilinearCorners3d<B: Backend> {
    /// Corner indices, already clamped into `[0, size)` so they are safe to gather and scatter
    /// with.
    index: [[IntTensor<B>; 2]; 3],
    /// Interpolation weights, `[1 - frac, frac]`.
    weight: [[FloatTensor<B>; 2]; 3],
    /// Whether the *unclamped* corner index was inside the volume. `Some` only under
    /// [`GridSamplePaddingMode::Zeros`], where clamping would otherwise silently sample — and, in
    /// the backward, deposit gradient on — a voxel the coordinate never reached.
    inside: Option<[[BoolTensor<B>; 2]; 3]>,
}

impl<B: Backend> TrilinearCorners3d<B> {
    /// Resolves a sampling grid into corner indices and weights.
    fn new(
        grid: FloatTensor<B>,
        sizes: [usize; 3],
        padding_mode: GridSamplePaddingMode,
        align_corners: bool,
    ) -> Self {
        let device = grid.device();
        let settings = get_device_settings::<B>(&device);
        let [d_in, h_in, w_in] = sizes;
        // Per-axis input extents, ordered `x, y, z` to match `index` / `weight`.
        let extent = [w_in, h_in, d_in];

        let coords = grid_sample_3d_coords::<B>(grid, sizes, padding_mode, align_corners);
        let floored = coords.clone().map(B::float_floor);

        // Convert to integer indices; index 0 is the lower corner, index 1 the upper one.
        let unclamped =
            [0usize, 1, 2].map(|a| corner_indices::<B>(floored[a].clone(), settings.int_dtype));

        // Create masks for out-of-bounds coordinates (only used for zeros padding)
        let inside = (padding_mode == GridSamplePaddingMode::Zeros).then(|| {
            [0usize, 1, 2].map(|a| {
                unclamped[a]
                    .clone()
                    .map(|i| in_bounds::<B>(i, extent[a], settings.bool_dtype))
            })
        });

        // Clamp indices to valid range for gather
        let index = [0usize, 1, 2].map(|a| {
            let max = (extent[a] - 1) as i32;
            unclamped[a]
                .clone()
                .map(|i| B::int_clamp(i, 0.into(), max.into()))
        });

        // Trilinear weights per axis: index 0 pairs with the lower corner, index 1 with the upper.
        let weight = [0usize, 1, 2].map(|a| {
            let frac = B::float_sub(coords[a].clone(), floored[a].clone());
            [
                B::float_add_scalar(B::float_neg(frac.clone()), 1f32.into()),
                frac,
            ]
        });

        Self {
            index,
            weight,
            inside,
        }
    }

    /// The clamped `[x, y, z]` indices of one corner.
    fn corner_index(&self, corner: [usize; 3]) -> [IntTensor<B>; 3] {
        [0usize, 1, 2].map(|a| self.index[a][corner[a]].clone())
    }

    /// The product of the three per-axis interpolation weights of one corner.
    fn corner_weight(&self, corner: [usize; 3]) -> FloatTensor<B> {
        B::float_mul(
            B::float_mul(
                self.weight[0][corner[0]].clone(),
                self.weight[1][corner[1]].clone(),
            ),
            self.weight[2][corner[2]].clone(),
        )
    }

    /// The product of the two per-axis weights *other than* `axis`.
    ///
    /// This is the factor multiplying `d w_axis / d q_axis = ±1` in the grid gradient: the corner
    /// weight is separable, so differentiating it along one axis leaves the other two untouched.
    fn cross_weight(&self, corner: [usize; 3], axis: usize) -> FloatTensor<B> {
        let [first, second] = match axis {
            0 => [1usize, 2],
            1 => [0usize, 2],
            _ => [0usize, 1],
        };

        B::float_mul(
            self.weight[first][corner[first]].clone(),
            self.weight[second][corner[second]].clone(),
        )
    }

    /// Zeroes the entries of `tensor` whose corner fell outside the volume.
    ///
    /// Only `Zeros` padding has such corners; the other modes keep every coordinate in range.
    /// `tensor` must have shape `shape`, which broadcasts the (N, 1, ...) mask across `C`.
    fn mask_outside(
        &self,
        tensor: FloatTensor<B>,
        corner: [usize; 3],
        shape: &Shape,
    ) -> FloatTensor<B> {
        let Some(inside) = &self.inside else {
            return tensor;
        };

        let corner_inside = B::bool_and(
            B::bool_and(inside[0][corner[0]].clone(), inside[1][corner[1]].clone()),
            inside[2][corner[2]].clone(),
        );
        let mask = B::bool_expand(B::bool_not(corner_inside), shape.clone());

        B::float_mask_fill(tensor, mask, 0f32.into())
    }
}

/// The single voxel each output position samples in nearest mode.
///
/// Returns the clamped `[x, y, z]` indices and, under `Zeros` padding, a (N, 1, D_out, H_out,
/// W_out) mask of the positions that landed outside the volume.
fn nearest_voxel_3d<B: Backend>(
    grid: FloatTensor<B>,
    sizes: [usize; 3],
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> ([IntTensor<B>; 3], Option<BoolTensor<B>>) {
    let device = grid.device();
    let settings = get_device_settings::<B>(&device);
    let [d_in, h_in, w_in] = sizes;
    let extent = [w_in, h_in, d_in];

    let coords = grid_sample_3d_coords::<B>(grid, sizes, padding_mode, align_corners);
    let rounded = coords.map(|coord| B::float_into_int(B::float_round(coord), settings.int_dtype));

    let outside = (padding_mode == GridSamplePaddingMode::Zeros).then(|| {
        let [x, y, z] = [0usize, 1, 2]
            .map(|a| in_bounds::<B>(rounded[a].clone(), extent[a], settings.bool_dtype));
        B::bool_not(B::bool_and(B::bool_and(x, y), z))
    });

    let indices = [0usize, 1, 2].map(|a| {
        let max = (extent[a] - 1) as i32;
        B::int_clamp(rounded[a].clone(), 0.into(), max.into())
    });

    (indices, outside)
}

/// Split a 3-D sampling grid into its three pixel-coordinate planes.
///
/// `grid` has shape (N, D_out, H_out, W_out, 3) with the last dimension ordered `(x, y, z)`. The
/// returned tensors are `[x, y, z]`, each of shape (N, 1, D_out, H_out, W_out), unnormalized from
/// [-1, 1] into pixel coordinates and already adjusted for `padding_mode`. The singleton channel
/// axis lets them broadcast against the sampled tensor's `C`.
fn grid_sample_3d_coords<B: Backend>(
    grid: FloatTensor<B>,
    sizes: [usize; 3],
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> [FloatTensor<B>; 3] {
    let [d_in, h_in, w_in] = sizes;
    let extent = [w_in as f64, h_in as f64, d_in as f64];

    let mut coords = grid_sample_3d_raw_coords::<B>(grid, sizes, align_corners)
        .into_iter()
        .zip(extent)
        .map(|(coord, size)| pad_coordinate::<B>(coord, size, padding_mode, align_corners));

    let x = coords.next().expect("Three coordinate axes");
    let y = coords.next().expect("Three coordinate axes");
    let z = coords.next().expect("Three coordinate axes");
    [x, y, z]
}

/// The `[x, y, z]` pixel coordinates of a sampling grid *before* any padding mode is applied.
///
/// Each is (N, 1, D_out, H_out, W_out). The backward needs these unpadded coordinates because the
/// padding derivative is a function of where the coordinate was, not of where it was moved to.
fn grid_sample_3d_raw_coords<B: Backend>(
    grid: FloatTensor<B>,
    [d_in, h_in, w_in]: [usize; 3],
    align_corners: bool,
) -> [FloatTensor<B>; 3] {
    let n = grid.shape()[0];
    let d_out = grid.shape()[1];
    let h_out = grid.shape()[2];
    let w_out = grid.shape()[3];

    let coord_shape = Shape::new([n, 1, d_out, h_out, w_out]);
    let axis_slice = |axis: isize| {
        vec![
            Slice::new(0, Some(n as isize), 1),
            Slice::new(0, Some(d_out as isize), 1),
            Slice::new(0, Some(h_out as isize), 1),
            Slice::new(0, Some(w_out as isize), 1),
            Slice::new(axis, Some(axis + 1), 1),
        ]
    };

    // Each axis is scaled by its own input extent: x by W_in, y by H_in and z by D_in.
    let extent = [w_in as f64, h_in as f64, d_in as f64];

    [0usize, 1, 2].map(|axis| {
        // Separate x, y and z coordinates from grid; shape: (N, D_out, H_out, W_out, 1)
        let coord = B::float_slice(grid.clone(), &axis_slice(axis as isize));
        let coord = B::float_reshape(coord, coord_shape.clone());

        // Convert normalized grid coordinates [-1, 1] to pixel coordinates.
        let coord = B::float_add_scalar(coord, 1f32.into());
        let coord = B::float_mul_scalar(coord, coord_scale(extent[axis], align_corners).into());

        if align_corners {
            // align_corners=true: x_pixel = (x_norm + 1) * (width - 1) / 2
            // Maps -1 to 0 and 1 to width - 1
            coord
        } else {
            // align_corners=false: x_pixel = (x_norm + 1) * width / 2 - 0.5
            // Maps -1 to -0.5 and 1 to width - 0.5
            B::float_sub_scalar(coord, 0.5f32.into())
        }
    })
}

/// The `[-1, 1]` to pixel scale factor of one axis, which is also `d(pixel)/d(grid value)`.
fn coord_scale(size: f64, align_corners: bool) -> f64 {
    if align_corners {
        (size - 1.0) / 2.0
    } else {
        size / 2.0
    }
}

/// Applies a padding mode to one axis' pixel coordinates.
fn pad_coordinate<B: Backend>(
    coord: FloatTensor<B>,
    size: f64,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> FloatTensor<B> {
    match padding_mode {
        // Clamp coordinates to valid range [0, size-1]
        GridSamplePaddingMode::Border => {
            B::float_clamp(coord, 0f32.into(), ((size - 1.0) as f32).into())
        }
        // Reflect coordinates at boundaries
        GridSamplePaddingMode::Reflection => reflect_coordinates::<B>(coord, size, align_corners),
        // Keep coordinates as-is, out-of-bounds corners are masked later
        GridSamplePaddingMode::Zeros => coord,
    }
}

/// `d(padded coordinate) / d(raw coordinate)` for one axis, or `None` when it is identically one.
///
/// `coord` is the *raw*, pre-padding pixel coordinate, i.e. the output of
/// [`grid_sample_3d_raw_coords`].
///
/// `Border` treats both borders themselves as out of bounds, so gradient flows only strictly
/// inside `(0, size - 1)`. That is PyTorch's `clip_coordinates_set_grad` rule verbatim, and it is
/// what keeps a coordinate parked on the edge from receiving a phantom gradient.
///
/// `Reflection` differentiates the triangle wave of [`reflect_coordinates`]:
/// `d/dp [span - |x_mod - span| + min] = -sign(x_mod - span) * sign(p - min)`. The two comparisons
/// below are written so the ties fall PyTorch's way — at `p == min` the reflection sign is `+1`
/// (`reflect_coordinates_set_grad`'s `else` branch) and at `x_mod == span` the flip sign is `-1`
/// (its odd-`flips` branch). PyTorch then clips the reflected coordinate with the same
/// borders-are-outside rule as `Border` (`clip_coordinates_set_grad`), so the gradient is also
/// zero wherever the reflected coordinate lands at or beyond `0` or `size - 1`.
fn pad_coordinate_grad<B: Backend>(
    coord: FloatTensor<B>,
    size: f64,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> Option<FloatTensor<B>> {
    let device = coord.device();
    let float_dtype: FloatDType = coord.dtype().into();
    let bool_dtype = get_device_settings::<B>(&device).bool_dtype;

    match padding_mode {
        GridSamplePaddingMode::Zeros => None,
        GridSamplePaddingMode::Border => {
            let inside = B::bool_and(
                B::float_greater_elem(coord.clone(), 0f32.into(), bool_dtype),
                B::float_lower_elem(coord, ((size - 1.0) as f32).into(), bool_dtype),
            );
            Some(B::bool_into_float(inside, float_dtype))
        }
        GridSamplePaddingMode::Reflection => {
            let (min_val, max_val) = if align_corners {
                (0.0f32, (size - 1.0) as f32)
            } else {
                (-0.5f32, (size - 0.5) as f32)
            };

            let span = max_val - min_val;
            if span <= 0.0 {
                // Edge case: size is 1, so the forward is constant and nothing flows back.
                return Some(B::float_mul_scalar(coord, 0f32.into()));
            }

            let period = 2.0 * span;
            let shape = coord.shape();

            // The clip that follows the reflection: no gradient at or beyond either border.
            let reflected = reflect_coordinates::<B>(coord.clone(), size, align_corners);
            let inside = B::bool_and(
                B::float_greater_elem(reflected.clone(), 0f32.into(), bool_dtype),
                B::float_lower_elem(reflected, ((size - 1.0) as f32).into(), bool_dtype),
            );

            // offset = coord - min; x_mod = |offset| mod period
            let offset = B::float_sub_scalar(coord, min_val.into());
            let x = B::float_abs(offset.clone());
            let x_div_floor = B::float_floor(B::float_div_scalar(x.clone(), period.into()));
            let x_mod = B::float_sub(x, B::float_mul_scalar(x_div_floor, period.into()));

            let ones = B::float_ones(shape, &device, float_dtype);
            let below_min = B::float_lower_elem(offset, 0f32.into(), bool_dtype);
            let reflect_sign = B::float_mask_fill(ones.clone(), below_min, (-1f32).into());
            let flipped = B::float_greater_equal_elem(x_mod, span.into(), bool_dtype);
            let flip_sign = B::float_mask_fill(ones, flipped, (-1f32).into());

            let sign = B::float_mul(reflect_sign, flip_sign);
            Some(B::float_mul(sign, B::bool_into_float(inside, float_dtype)))
        }
    }
}

/// The two integer corner indices bracketing a floored coordinate: `[floor, floor + 1]`.
fn corner_indices<B: Backend>(floored: FloatTensor<B>, int_dtype: IntDType) -> [IntTensor<B>; 2] {
    [
        B::float_into_int(floored.clone(), int_dtype),
        B::float_into_int(B::float_add_scalar(floored, 1f32.into()), int_dtype),
    ]
}

/// Whether each index falls inside `[0, size)`.
fn in_bounds<B: Backend>(idx: IntTensor<B>, size: usize, bool_dtype: BoolDType) -> BoolTensor<B> {
    B::bool_and(
        B::int_greater_equal_elem(idx.clone(), 0.into(), bool_dtype),
        B::int_lower_elem(idx, (size as i32).into(), bool_dtype),
    )
}

/// Gather one voxel per output position from a spatially flattened tensor.
///
/// `tensor_flat` is (N, C, D_in * H_in * W_in) and the `[x, y, z]` indices are (N, 1, ...). The
/// result is reshaped to `out_shape`.
fn gather_voxels<B: Backend>(
    tensor_flat: FloatTensor<B>,
    indices: [IntTensor<B>; 3],
    [h_in, w_in]: [usize; 2],
    [n, c, spatial_out]: [usize; 3],
    out_shape: Shape,
) -> FloatTensor<B> {
    let idx = voxel_indices::<B>(indices, [h_in, w_in], [n, c, spatial_out]);
    let sample = B::float_gather(2, tensor_flat, idx);

    B::float_reshape(sample, out_shape)
}

/// The linear offset `(z * H_in + y) * W_in + x` of one voxel per output position, broadcast
/// across the channel axis.
///
/// The `[x, y, z]` indices are (N, 1, D_out, H_out, W_out); the result is
/// (N, C, D_out * H_out * W_out), ready to index the spatial axis of a flattened volume with
/// `float_gather` or `float_scatter`.
fn voxel_indices<B: Backend>(
    [x, y, z]: [IntTensor<B>; 3],
    [h_in, w_in]: [usize; 2],
    [n, c, spatial_out]: [usize; 3],
) -> IntTensor<B> {
    let idx = B::int_mul_scalar(z, (h_in as i32).into());
    let idx = B::int_add(idx, y);
    let idx = B::int_mul_scalar(idx, (w_in as i32).into());
    let idx = B::int_add(idx, x);

    // [N, 1, D_out, H_out, W_out] -> [N, 1, D_out * H_out * W_out] -> [N, C, spatial]
    let idx = B::int_reshape(idx, Shape::new([n, 1, spatial_out]));

    B::int_expand(idx, Shape::new([n, c, spatial_out]))
}
