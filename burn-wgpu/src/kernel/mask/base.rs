use crate::{element::WgpuElement, tensor::WgpuTensor, JitRuntime};

/// Execute the mask fill kernel.
pub fn mask_fill<B: JitRuntime, E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<B, E, D>,
    mask: WgpuTensor<B, u32, D>,
    value: E,
) -> WgpuTensor<B, E, D> {
    if tensor.can_mut() {
        return super::mask_fill::mask_fill_inplace(tensor, mask, value);
    }

    super::mask_fill::mask_fill(tensor, mask, value)
}

/// Execute the mask where kernel.
pub fn mask_where<B: JitRuntime, E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<B, E, D>,
    mask: WgpuTensor<B, u32, D>,
    value: WgpuTensor<B, E, D>,
) -> WgpuTensor<B, E, D> {
    if tensor.can_mut_broadcast(&value) {
        return super::mask_where::mask_where_inplace(tensor, mask, value, false);
    }
    if value.can_mut_broadcast(&tensor) {
        return super::mask_where::mask_where_inplace(value, mask, tensor, true);
    }

    super::mask_where::mask_where(tensor, mask, value)
}
