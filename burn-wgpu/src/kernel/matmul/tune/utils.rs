use burn_tensor::Shape;

use crate::element::WgpuElement;

pub(crate) fn n_bytes<E, const D: usize>(shape: &Shape<D>) -> usize {
    shape.num_elements() * core::mem::size_of::<E>()
}

pub(crate) fn reduce_shape<const D: usize>(shape: &Shape<D>) -> Shape<3> {
    let n_batches = 2;
    Shape::new([n_batches, shape.dims[D - 2], shape.dims[D - 1]])
}

pub(crate) fn fill_bytes<E: WgpuElement, const D: usize>(value: u8, shape: &Shape<D>) -> Vec<u8> {
    vec![value; n_bytes::<E, D>(shape)]
}
