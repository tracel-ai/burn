use crate::ElementConversion;
use crate::backend::Backend;
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
pub fn affine_grid_2d<B: Backend>(transform: Tensor<B, 3>, dims: [usize; 4]) -> Tensor<B, 4> {
    let [batch_size, _c, height, width] = dims;

    let device = &transform.device();

    let x = Tensor::<B, 1, Int>::arange(0..width as i64, device)
        .reshape([1, width])
        .expand([height, width]);
    let y = Tensor::<B, 1, Int>::arange(0..height as i64, device)
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
