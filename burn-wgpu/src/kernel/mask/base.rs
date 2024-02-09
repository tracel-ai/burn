use crate::{element::WgpuElement, tensor::WgpuTensor, Runtime};

/// Execute the mask fill kernel.
pub fn mask_fill<R: Runtime, E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<R, E, D>,
    mask: WgpuTensor<R, u32, D>,
    value: E,
) -> WgpuTensor<R, E, D> {
    if tensor.can_mut() {
        return super::mask_fill::mask_fill_inplace(tensor, mask, value);
    }

    super::mask_fill::mask_fill(tensor, mask, value)
}

/// Execute the mask where kernel.
pub fn mask_where<R: Runtime, E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<R, E, D>,
    mask: WgpuTensor<R, u32, D>,
    value: WgpuTensor<R, E, D>,
) -> WgpuTensor<R, E, D> {
    if tensor.can_mut_broadcast(&value) {
        return super::mask_where::mask_where_inplace(tensor, mask, value, false);
    }
    if value.can_mut_broadcast(&tensor) {
        return super::mask_where::mask_where_inplace(value, mask, tensor, true);
    }

    super::mask_where::mask_where(tensor, mask, value)
}
