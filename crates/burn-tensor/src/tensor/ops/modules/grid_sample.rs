use crate::{ElementConversion, Shape, Slice, TensorMetadata, backend::Backend, ops::FloatTensor};
use alloc::vec;

/// Default implementation of float_grid_sample_2d with bilinear interpolation and border padding
///
/// # Arguments
///
/// * `tensor` - The tensor being sampled from, shape (N, C, H_in, W_in)
/// * `grid` - A tensor of locations, with shape (N, H_out, W_out, 2). Values are [-1, 1].
///   A [x = -1, y = -1] means top-left, and [x = 1, y = 1] means bottom-right
///
/// # Returns
///
/// A tensor with shape (N, C, H_out, W_out)
pub fn float_grid_sample_2d_bilinear<B: Backend>(
    tensor: FloatTensor<B>,
    grid: FloatTensor<B>,
) -> FloatTensor<B> {
    let n = tensor.shape().dims[0];
    let c = tensor.shape().dims[1];
    let h_in = tensor.shape().dims[2];
    let w_in = tensor.shape().dims[3];
    let h_out = grid.shape().dims[1];
    let w_out = grid.shape().dims[2];

    let x_max_half = (w_in - 1) as f64 / 2.0;
    let y_max_half = (h_in - 1) as f64 / 2.0;

    // Clamp grid
    let grid = B::float_clamp(grid, (-1_f32).elem(), (1_f32).elem());

    // Separate x and y coordinates
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

    // Scale grid locations from [-1, 1] and [-1, 1] to [0..W_out] and [0..H_out]
    let grid_x = B::float_mul_scalar(grid_x, x_max_half.elem());
    let grid_x = B::float_add_scalar(grid_x, x_max_half.elem());
    let grid_y = B::float_mul_scalar(grid_y, x_max_half.elem());
    let grid_y = B::float_add_scalar(grid_y, y_max_half.elem());

    // Get low and high x locations
    let grid_x_floored = B::float_floor(grid_x.clone());
    let grid_x_plus_one = B::float_floor(B::float_add_scalar(grid_x.clone(), 1.elem()));
    let x_indices_low = B::float_into_int(grid_x_floored.clone());
    let x_indices_high = B::float_into_int(grid_x_plus_one.clone());

    // Get low and high x locations
    let grid_y_floored = B::float_floor(grid_y.clone());
    let grid_y_plus_one = B::float_floor(B::float_add_scalar(grid_y.clone(), 1.elem()));
    let y_indices_low = B::float_into_int(grid_y_floored.clone());
    let y_indices_high = B::float_into_int(grid_y_plus_one.clone());

    // Clamp locations: border padding
    let x_indices_low = B::int_clamp(x_indices_low, 0.elem(), ((w_in - 1) as u32).elem());
    let x_indices_high = B::int_clamp(x_indices_high, 0.elem(), ((w_in - 1) as u32).elem());
    let y_indices_low = B::int_clamp(y_indices_low, 0.elem(), ((h_in - 1) as u32).elem());
    let y_indices_high = B::int_clamp(y_indices_high, 0.elem(), ((h_in - 1) as u32).elem());

    // Needs shape (N, C, H_out, W_out, W_in) for the first gather operationd
    let y_indices_low = B::int_reshape(y_indices_low, Shape::new([n, 1, h_out, w_out, 1]));
    let y_indices_low = B::int_expand(y_indices_low, Shape::new([n, c, h_out, w_out, w_in]));
    let y_indices_high = B::int_reshape(y_indices_high, Shape::new([n, 1, h_out, w_out, 1]));
    let y_indices_high = B::int_expand(y_indices_high, Shape::new([n, c, h_out, w_out, w_in]));

    // Needs shape (N, C, H_out, W_out, 1) for the second gather operation
    let x_indices_low = B::int_reshape(x_indices_low, Shape::new([n, 1, h_out, w_out, 1]));
    let x_indices_low = B::int_expand(x_indices_low, Shape::new([n, c, h_out, w_out, 1]));
    let x_indices_high = B::int_reshape(x_indices_high, Shape::new([n, 1, h_out, w_out, 1]));
    let x_indices_high = B::int_expand(x_indices_high, Shape::new([n, c, h_out, w_out, 1]));

    // Reshape tensor for gather operation
    let tensor = B::float_reshape(tensor, Shape::new([n, c, h_in, 1, w_in]));
    let tensor = B::float_expand(tensor, Shape::new([n, c, h_in, w_out, w_in]));

    // Gather on x and y. Watch out for the shapes
    let sample_00 = B::float_gather(2, tensor.clone(), y_indices_low.clone());
    let sample_00 = B::float_gather(4, sample_00, x_indices_low.clone());

    let sample_01 = B::float_gather(2, tensor.clone(), y_indices_high.clone());
    let sample_01 = B::float_gather(4, sample_01, x_indices_low.clone());

    let sample_10 = B::float_gather(2, tensor.clone(), y_indices_low.clone());
    let sample_10 = B::float_gather(4, sample_10, x_indices_high.clone());

    let sample_11 = B::float_gather(2, tensor, y_indices_high);
    let sample_11 = B::float_gather(4, sample_11, x_indices_high);

    // Reshape to (N, C, H_out, W_out) for multiplying with weights
    let sample_00 = B::float_reshape(sample_00, Shape::new([n, c, h_out, w_out]));
    let sample_01 = B::float_reshape(sample_01, Shape::new([n, c, h_out, w_out]));
    let sample_10 = B::float_reshape(sample_10, Shape::new([n, c, h_out, w_out]));
    let sample_11 = B::float_reshape(sample_11, Shape::new([n, c, h_out, w_out]));

    // Weights for bilinear interp
    let weight_00 = B::float_mul(
        B::float_sub(grid_x_plus_one.clone(), grid_x.clone()),
        B::float_sub(grid_y_plus_one.clone(), grid_y.clone()),
    );
    let weight_10 = B::float_mul(
        B::float_sub(grid_x.clone(), grid_x_floored.clone()),
        B::float_sub(grid_y_plus_one.clone(), grid_y.clone()),
    );
    let weight_01 = B::float_mul(
        B::float_sub(grid_x_plus_one.clone(), grid_x.clone()),
        B::float_sub(grid_y.clone(), grid_y_floored.clone()),
    );
    let weight_11 = B::float_mul(
        B::float_sub(grid_x.clone(), grid_x_floored),
        B::float_sub(grid_y.clone(), grid_y_floored),
    );

    // Bilinear interp
    let sample_0 = B::float_add(
        B::float_mul(sample_00, weight_00),
        B::float_mul(sample_01, weight_01),
    );
    let sample_1 = B::float_add(
        B::float_mul(sample_10, weight_10),
        B::float_mul(sample_11, weight_11),
    );
    B::float_add(sample_0, sample_1)
}
