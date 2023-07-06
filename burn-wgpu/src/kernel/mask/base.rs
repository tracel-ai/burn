use crate::{element::WgpuElement, tensor::WgpuTensor};

/// Execute the mask fill kernel.
pub fn mask_fill<E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: E,
) -> WgpuTensor<E, D> {
    if tensor.can_mut() {
        return super::mask_fill::mask_fill_inplace(tensor, mask, value);
    }

    super::mask_fill::mask_fill(tensor, mask, value)
}

/// Execute the mask where kernel.
pub fn mask_where<E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    if tensor.can_mut_broadcast(&value) {
        return super::mask_where::mask_where_inplace(tensor, mask, value, false);
    }
    if value.can_mut_broadcast(&tensor) {
        return super::mask_where::mask_where_inplace(value, mask, tensor, true);
    }

    super::mask_where::mask_where(tensor, mask, value)
}
