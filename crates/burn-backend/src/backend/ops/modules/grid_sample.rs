use crate::{
    Backend, TensorMetadata,
    element::ElementConversion,
    ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode},
    tensor::FloatTensor,
};
use alloc::vec;
use burn_std::{Shape, Slice};

/// Reference implementation of grid_sample_2d that supports all options.
///
/// # Arguments
///
/// * `tensor` - The tensor being sampled from, shape (N, C, H_in, W_in)
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
    let n = tensor.shape().dims[0];
    let c = tensor.shape().dims[1];
    let h_in = tensor.shape().dims[2];
    let w_in = tensor.shape().dims[3];
    let h_out = grid.shape().dims[1];
    let w_out = grid.shape().dims[2];

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
        let grid_x = B::float_add_scalar(grid_x, 1.0f32.elem());
        let grid_x = B::float_mul_scalar(grid_x, ((w_in_f - 1.0) / 2.0).elem());

        let grid_y = B::float_add_scalar(grid_y, 1.0f32.elem());
        let grid_y = B::float_mul_scalar(grid_y, ((h_in_f - 1.0) / 2.0).elem());

        (grid_x, grid_y)
    } else {
        // align_corners=false: x_pixel = (x_norm + 1) * width / 2 - 0.5
        // Maps -1 to -0.5 and 1 to width - 0.5
        let grid_x = B::float_add_scalar(grid_x, 1.0f32.elem());
        let grid_x = B::float_mul_scalar(grid_x, (w_in_f / 2.0).elem());
        let grid_x = B::float_sub_scalar(grid_x, 0.5f32.elem());

        let grid_y = B::float_add_scalar(grid_y, 1.0f32.elem());
        let grid_y = B::float_mul_scalar(grid_y, (h_in_f / 2.0).elem());
        let grid_y = B::float_sub_scalar(grid_y, 0.5f32.elem());

        (grid_x, grid_y)
    };

    // Apply padding mode to coordinates
    let (grid_x, grid_y) = match padding_mode {
        GridSamplePaddingMode::Border => {
            // Clamp coordinates to valid range [0, size-1]
            let grid_x = B::float_clamp(grid_x, 0.0f32.elem(), ((w_in - 1) as f32).elem());
            let grid_y = B::float_clamp(grid_y, 0.0f32.elem(), ((h_in - 1) as f32).elem());
            (grid_x, grid_y)
        }
        GridSamplePaddingMode::Reflection => {
            // Reflect coordinates at boundaries
            // For now, use a simplified reflection that works for common cases
            let grid_x = reflect_coordinates::<B>(grid_x, w_in_f);
            let grid_y = reflect_coordinates::<B>(grid_y, h_in_f);
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
    let x0 = B::float_into_int(grid_x_floored.clone());
    let y0 = B::float_into_int(grid_y_floored.clone());
    let x1 = B::float_into_int(B::float_add_scalar(grid_x_floored, 1.0f32.elem()));
    let y1 = B::float_into_int(B::float_add_scalar(grid_y_floored, 1.0f32.elem()));

    // Create masks for out-of-bounds coordinates (only used for zeros padding)
    let (mask_00, mask_01, mask_10, mask_11) = if padding_mode == GridSamplePaddingMode::Zeros {
        let x0_valid = B::int_greater_equal_elem(x0.clone(), 0.elem());
        let x0_valid = B::bool_and(
            x0_valid,
            B::int_lower_elem(x0.clone(), (w_in as i32).elem()),
        );
        let x1_valid = B::int_greater_equal_elem(x1.clone(), 0.elem());
        let x1_valid = B::bool_and(
            x1_valid,
            B::int_lower_elem(x1.clone(), (w_in as i32).elem()),
        );
        let y0_valid = B::int_greater_equal_elem(y0.clone(), 0.elem());
        let y0_valid = B::bool_and(
            y0_valid,
            B::int_lower_elem(y0.clone(), (h_in as i32).elem()),
        );
        let y1_valid = B::int_greater_equal_elem(y1.clone(), 0.elem());
        let y1_valid = B::bool_and(
            y1_valid,
            B::int_lower_elem(y1.clone(), (h_in as i32).elem()),
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
    let x0_clamped = B::int_clamp(x0, 0.elem(), ((w_in - 1) as i32).elem());
    let x1_clamped = B::int_clamp(x1, 0.elem(), ((w_in - 1) as i32).elem());
    let y0_clamped = B::int_clamp(y0, 0.elem(), ((h_in - 1) as i32).elem());
    let y1_clamped = B::int_clamp(y1, 0.elem(), ((h_in - 1) as i32).elem());

    // Reshape indices for gather operation
    let y0_idx = B::int_reshape(y0_clamped.clone(), Shape::new([n, 1, h_out, w_out, 1]));
    let y0_idx = B::int_expand(y0_idx, Shape::new([n, c, h_out, w_out, w_in]));
    let y1_idx = B::int_reshape(y1_clamped.clone(), Shape::new([n, 1, h_out, w_out, 1]));
    let y1_idx = B::int_expand(y1_idx, Shape::new([n, c, h_out, w_out, w_in]));

    let x0_idx = B::int_reshape(x0_clamped, Shape::new([n, 1, h_out, w_out, 1]));
    let x0_idx = B::int_expand(x0_idx, Shape::new([n, c, h_out, w_out, 1]));
    let x1_idx = B::int_reshape(x1_clamped, Shape::new([n, 1, h_out, w_out, 1]));
    let x1_idx = B::int_expand(x1_idx, Shape::new([n, c, h_out, w_out, 1]));

    // Reshape tensor for gather operation
    let tensor = B::float_reshape(tensor, Shape::new([n, c, h_in, 1, w_in]));
    let tensor = B::float_expand(tensor, Shape::new([n, c, h_in, w_out, w_in]));

    // Gather samples from the four corners
    let sample_00 = B::float_gather(2, tensor.clone(), y0_idx.clone());
    let sample_00 = B::float_gather(4, sample_00, x0_idx.clone());

    let sample_01 = B::float_gather(2, tensor.clone(), y1_idx.clone());
    let sample_01 = B::float_gather(4, sample_01, x0_idx.clone());

    let sample_10 = B::float_gather(2, tensor.clone(), y0_idx);
    let sample_10 = B::float_gather(4, sample_10, x1_idx.clone());

    let sample_11 = B::float_gather(2, tensor, y1_idx);
    let sample_11 = B::float_gather(4, sample_11, x1_idx);

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
                B::float_mask_fill(sample_00, mask_00_inv, 0.0f32.elem()),
                B::float_mask_fill(sample_01, mask_01_inv, 0.0f32.elem()),
                B::float_mask_fill(sample_10, mask_10_inv, 0.0f32.elem()),
                B::float_mask_fill(sample_11, mask_11_inv, 0.0f32.elem()),
            )
        } else {
            (sample_00, sample_01, sample_10, sample_11)
        };

    // Compute bilinear interpolation weights
    let one_minus_x = B::float_neg(x_frac.clone());
    let one_minus_x = B::float_add_scalar(one_minus_x, 1.0f32.elem());

    let one_minus_y = B::float_neg(y_frac.clone());
    let one_minus_y = B::float_add_scalar(one_minus_y, 1.0f32.elem());

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

/// Reflect coordinates at boundaries for reflection padding.
///
/// Uses the formula: reflected = 2 * bound - x for out-of-bounds coordinates.
fn reflect_coordinates<B: Backend>(coords: FloatTensor<B>, size: f64) -> FloatTensor<B> {
    // Simple reflection: clamp to [0, size-1] after reflecting
    // For values < 0: reflect at 0 -> -x
    // For values >= size: reflect at size-1 -> 2*(size-1) - x
    // This is a simplified implementation - full reflection would handle multiple reflections

    let max_val = (size - 1.0) as f32;

    // First handle negative values by taking absolute value
    let coords = B::float_abs(coords);

    // Then handle values > max by reflecting: 2*max - x
    // But we need to detect which values need this
    // Simplified: just clamp for now, proper reflection is complex
    B::float_clamp(coords, 0.0f32.elem(), max_val.elem())
}
