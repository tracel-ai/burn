use burn_tensor::Element;

use crate::{element::WgpuElement, ops::numeric::ones_device, tensor::WgpuTensor};

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
