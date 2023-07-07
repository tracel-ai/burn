use crate::{element::WgpuElement, tensor::WgpuTensor};
use burn_tensor::Shape;

pub(crate) fn shape_out<E: WgpuElement, const D: usize>(
    lhs: &WgpuTensor<E, D>,
    rhs: &WgpuTensor<E, D>,
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
