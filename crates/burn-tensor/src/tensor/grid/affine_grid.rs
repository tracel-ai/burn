use crate::ElementConversion;
use crate::backend::Backend;
use crate::tensor::{Int, Tensor};

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
pub fn affine_grid_2d<B: Backend>(transform: Tensor<B, 3>, dims: [usize; 4]) -> Tensor<B, 4> {
    let batch_size = dims[0];
    let _channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    let device = &transform.device();

    let x = Tensor::<B, 1, Int>::arange(0..width as i64, device)
        .reshape([1, width, 1])
        .expand([height, width, 1]);
    let y = Tensor::<B, 1, Int>::arange(0..height as i64, device)
        .reshape([height, 1, 1])
        .expand([height, width, 1]);

    // from ints (0..(width-1)) and (0..(height-1)), to (-1.0..1.0)
    let x = x
        .float()
        .div_scalar(((width - 1) as f32 / 2.0).elem::<f32>())
        .sub_scalar((1_f32).elem::<f32>());
    let y = y
        .float()
        .div_scalar(((height - 1) as f32 / 2.0).elem::<f32>())
        .sub_scalar((1_f32).elem::<f32>());
    let w = Tensor::<B, 3>::ones([height, width, 1], device);

    let grid = Tensor::cat(vec![x, y, w], 2);

    // Arange coordinates as column vectors for transformation
    let transform = Tensor::cat(
        vec![
            transform,
            Tensor::<B, 3>::from([[[0, 0, 1]]]).expand([batch_size, 1, 3]),
        ],
        1,
    );
    let transform = transform
        .reshape([batch_size, 1, 1, 3, 3])
        .expand([batch_size, height, width, 3, 3]);
    let grid = grid
        .reshape([1, height, width, 3, 1])
        .expand([batch_size, height, width, 3, 1]);
    let grid = transform.matmul(grid.clone());
    let grid = grid.reshape([batch_size, height, width, 3]);

    // Homogeneous coordinates (x, y, w) to normal coordinates (x/w, y/w)
    let mut grid = grid.split_with_sizes(vec![2, 1], 3);
    let grid_xy = grid.remove(0);
    let grid_w = grid.remove(0);
    let grid_w = grid_w.expand([batch_size, height, width, 2]);

    grid_xy.div(grid_w)
}
