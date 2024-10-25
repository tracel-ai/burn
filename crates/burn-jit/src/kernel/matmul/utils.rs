use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};
use burn_tensor::Shape;

/// Creates an empty output tensor with matmul output shape
pub fn init_matmul_output<R: JitRuntime, E: JitElement>(
    lhs: &JitTensor<R, E>,
    rhs: &JitTensor<R, E>,
) -> JitTensor<R, E> {
    empty_device(lhs.client.clone(), lhs.device.clone(), shape_out(lhs, rhs))
}

pub(crate) fn shape_out<R: JitRuntime, E: JitElement>(
    lhs: &JitTensor<R, E>,
    rhs: &JitTensor<R, E>,
) -> Shape {
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
