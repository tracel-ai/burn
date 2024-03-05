use super::{mask_where::MaskWhereStrategy, MaskFillStrategy};
use crate::{element::JitElement, tensor::JitTensor, Runtime};

/// Execute the mask fill kernel.
pub(crate) fn mask_fill_auto<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    mask: JitTensor<R, u32, D>,
    value: E,
) -> JitTensor<R, E, D> {
    let strategy = if tensor.can_mut() {
        MaskFillStrategy::Inplace
    } else {
        MaskFillStrategy::Readonly
    };

    super::mask_fill(tensor, mask, value, strategy)
}

/// Execute the mask where kernel.
pub(crate) fn mask_where_auto<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    mask: JitTensor<R, u32, D>,
    value: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let strategy = if tensor.can_mut_broadcast(&value) {
        MaskWhereStrategy::InplaceLhs
    } else if value.can_mut_broadcast(&tensor) {
        MaskWhereStrategy::InplaceRhs
    } else {
        MaskWhereStrategy::Readonly
    };

    super::mask_where(tensor, mask, value, strategy)
}
