use burn_tensor::{Element, Shape};

use crate::{element::WgpuElement, ops::numeric::ones_device, tensor::WgpuTensor};

pub(crate) fn n_bytes<E, const D: usize>(shape: &Shape<D>) -> usize {
    shape.num_elements() * core::mem::size_of::<E>()
}

pub(crate) fn autotune_tensors<E: WgpuElement + Element, const D: usize>(
    tensor: &WgpuTensor<E, D>,
) -> WgpuTensor<E, 3> {
    let n_batches = 2;
    ones_device(
        tensor.client.clone(),
        tensor.device.clone(),
        [
            n_batches,
            tensor.shape.dims[D - 2],
            tensor.shape.dims[D - 1],
        ]
        .into(),
    )
}

pub(crate) fn fill_bytes<E: WgpuElement, const D: usize>(value: u8, shape: &Shape<D>) -> Vec<u8> {
    vec![value; n_bytes::<E, D>(shape)]
}
