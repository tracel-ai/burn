use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, Runtime};
use burn_tensor::Shape;

/// Creates an empty output tensor with matmul output shape
pub fn init_matmul_output<R: Runtime, E: JitElement, const D: usize>(
    lhs: &JitTensor<R, E, D>,
    rhs: &JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    empty_device(lhs.client.clone(), lhs.device.clone(), shape_out(lhs, rhs))
}

pub(crate) fn shape_out<R: Runtime, E: JitElement, const D: usize>(
    lhs: &JitTensor<R, E, D>,
    rhs: &JitTensor<R, E, D>,
) -> Shape<D> {
    let mut shape_out = [0; D];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });
    shape_out[D - 2] = lhs.shape.dims[D - 2];
    shape_out[D - 1] = rhs.shape.dims[D - 1];
    Shape::new(shape_out)
}
