use crate::{element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor, CubeRuntime};
use burn_tensor::Shape;

/// Creates an empty output tensor with matmul output shape
pub fn init_matmul_output<R: CubeRuntime, E: CubeElement>(
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
) -> CubeTensor<R> {
    empty_device::<R, E>(lhs.client.clone(), lhs.device.clone(), shape_out(lhs, rhs))
}

pub(crate) fn shape_out<R: CubeRuntime>(lhs: &CubeTensor<R>, rhs: &CubeTensor<R>) -> Shape {
    let ndims = lhs.shape.num_dims();
    let mut shape_out = vec![0; ndims];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });
    shape_out[ndims - 2] = lhs.shape.dims[ndims - 2];
    shape_out[ndims - 1] = rhs.shape.dims[ndims - 1];
    Shape::from(shape_out)
}
