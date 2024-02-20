use crate::{element::JitElement, tensor::JitTensor, Runtime};

/// Execute the mask fill kernel.
pub fn mask_fill<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    mask: JitTensor<R, u32, D>,
    value: E,
) -> JitTensor<R, E, D> {
    if tensor.can_mut() {
        return super::mask_fill::mask_fill_inplace(tensor, mask, value);
    }

    super::mask_fill::mask_fill(tensor, mask, value)
}

/// Execute the mask where kernel.
pub fn mask_where<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    mask: JitTensor<R, u32, D>,
    value: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    if tensor.can_mut_broadcast(&value) {
        return super::mask_where::mask_where_inplace(tensor, mask, value, false);
    }
    if value.can_mut_broadcast(&tensor) {
        return super::mask_where::mask_where_inplace(value, mask, tensor, true);
    }

    super::mask_where::mask_where(tensor, mask, value)
}
