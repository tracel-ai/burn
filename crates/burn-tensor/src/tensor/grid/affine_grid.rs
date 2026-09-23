use crate::Device;
use crate::ElementConversion;
use crate::s;
use crate::tensor::{Int, Tensor};
use alloc::vec;

/// Generate a tensor with homogeonous coordinates of each element's
/// transformed location
///
///
/// See:
///  - [torch.nn.functional.affine_grid](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html)
///
/// * `transform` - Transformation with shape (batch_size, 2, 3)
/// * `dims` - dimensions as (batch_size, channels, height, width)
///
/// # Returns
///
/// Tensor with shape (batch_size, height, width, 2), where dim 2 is (x, y)
/// All coordinates are broadcast on the batch dim
pub fn affine_grid_2d(transform: Tensor<3>, dims: [usize; 4]) -> Tensor<4> {
    let [batch_size, _c, height, width] = dims;

    let device = &transform.device();

    let x = Tensor::<1, Int>::arange(0..width as i64, device)
        .reshape([1, width])
        .expand([height, width]);
    let y = Tensor::<1, Int>::arange(0..height as i64, device)
        .reshape([height, 1])
        .expand([height, width]);

    // from ints (0..(width-1)) and (0..(height-1)), to (-1.0..1.0)
    let x = x
        .float()
        .div_scalar(((width - 1) as f32 / 2.0).elem::<f32>())
        .sub_scalar((1_f32).elem::<f32>());
    let y = y
        .float()
        .div_scalar(((height - 1) as f32 / 2.0).elem::<f32>())
        .sub_scalar((1_f32).elem::<f32>());

    // Broadcast to batch dimension
    let x = x.unsqueeze_dim::<3>(0).expand([batch_size, height, width]); // [B, H, W]
    let y = y.unsqueeze_dim::<3>(0).expand([batch_size, height, width]); // [B, H, W]

    // Apply affine transform
    let a_11 = transform.clone().slice(s![.., 0, 0]);
    let a_12 = transform.clone().slice(s![.., 0, 1]);
    let trans_x = transform.clone().slice(s![.., 0, 2]);

    let a_21 = transform.clone().slice(s![.., 1, 0]);
    let a_22 = transform.clone().slice(s![.., 1, 1]);
    let trans_y = transform.slice(s![.., 1, 2]);

    let grid_x = a_11.mul(x.clone()).add(a_12.mul(y.clone())).add(trans_x);
    let grid_y = a_21.mul(x).add(a_22.mul(y)).add(trans_y);

    Tensor::stack(vec![grid_x, grid_y], 3)
}

/// Generate a tensor with homogeonous coordinates of each element's
/// transformed location, in three dimensions
///
///
/// See:
///  - [torch.nn.functional.affine_grid](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html)
///
/// * `transform` - Transformation with shape (batch_size, 3, 4)
/// * `dims` - dimensions as (batch_size, channels, depth, height, width)
///
/// # Returns
///
/// Tensor with shape (batch_size, depth, height, width, 3), where dim 4 is (x, y, z) —
/// the reverse of the spatial dimension order, matching the layout
/// [`Tensor::grid_sample_3d`](crate::Tensor::grid_sample_3d) expects.
/// All coordinates are broadcast on the batch dim
///
/// Like [`affine_grid_2d`], the identity transform places -1 and 1 on the centers of the
/// corner voxels, so the grid pairs with `align_corners = true` when sampling. An axis of
/// extent one has a single center, at 0.
pub fn affine_grid_3d(transform: Tensor<3>, dims: [usize; 5]) -> Tensor<5> {
    let [batch_size, _c, depth, height, width] = dims;

    let device = &transform.device();

    // Normalized (-1.0..1.0) coordinates along each axis, laid out to broadcast over [D, H, W]
    let x = normalized_axis(width, device)
        .reshape([1, 1, width])
        .expand([depth, height, width]);
    let y = normalized_axis(height, device)
        .reshape([1, height, 1])
        .expand([depth, height, width]);
    let z = normalized_axis(depth, device)
        .reshape([depth, 1, 1])
        .expand([depth, height, width]);

    // Broadcast to batch dimension
    let x = x
        .unsqueeze_dim::<4>(0)
        .expand([batch_size, depth, height, width]); // [B, D, H, W]
    let y = y
        .unsqueeze_dim::<4>(0)
        .expand([batch_size, depth, height, width]); // [B, D, H, W]
    let z = z
        .unsqueeze_dim::<4>(0)
        .expand([batch_size, depth, height, width]); // [B, D, H, W]

    // Apply affine transform. Each coefficient is [B, 1, 1] after slicing; reshape it to
    // [B, 1, 1, 1] so it broadcasts against the [B, D, H, W] coordinate grids.
    let coefficient = |row: usize, col: usize| {
        transform
            .clone()
            .slice(s![.., row, col])
            .reshape([batch_size, 1, 1, 1])
    };

    let grid_x = coefficient(0, 0)
        .mul(x.clone())
        .add(coefficient(0, 1).mul(y.clone()))
        .add(coefficient(0, 2).mul(z.clone()))
        .add(coefficient(0, 3));
    let grid_y = coefficient(1, 0)
        .mul(x.clone())
        .add(coefficient(1, 1).mul(y.clone()))
        .add(coefficient(1, 2).mul(z.clone()))
        .add(coefficient(1, 3));
    let grid_z = coefficient(2, 0)
        .mul(x)
        .add(coefficient(2, 1).mul(y))
        .add(coefficient(2, 2).mul(z))
        .add(coefficient(2, 3));

    Tensor::stack(vec![grid_x, grid_y, grid_z], 4)
}

/// The `size` evenly spaced coordinates from -1.0 to 1.0 inclusive, i.e. the centers of the
/// voxels along one axis with `align_corners = true`.
///
/// A single voxel has no span to divide, so its center is 0 rather than the `0 / 0` the
/// general formula would produce.
fn normalized_axis(size: usize, device: &Device) -> Tensor<1> {
    if size <= 1 {
        return Tensor::zeros([size], device);
    }

    Tensor::<1, Int>::arange(0..size as i64, device)
        .float()
        .div_scalar(((size - 1) as f32 / 2.0).elem::<f32>())
        .sub_scalar((1_f32).elem::<f32>())
}
