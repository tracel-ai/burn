use super::{mask_where::MaskWhereStrategy, MaskFillStrategy};
use crate::{element::BasicJitElement, tensor::JitTensor, JitRuntime};

/// Execute the mask fill kernel.
pub(crate) fn mask_fill_auto<R: JitRuntime, E: BasicJitElement>(
    tensor: JitTensor<R>,
    mask: JitTensor<R>,
    value: E,
) -> JitTensor<R> {
    let strategy = if tensor.can_mut() {
        MaskFillStrategy::Inplace
    } else {
        MaskFillStrategy::Readonly
    };

    super::mask_fill(tensor, mask, value, strategy)
}

/// Execute the mask where kernel.
pub(crate) fn mask_where_auto<R: JitRuntime, E: BasicJitElement>(
    tensor: JitTensor<R>,
    mask: JitTensor<R>,
    value: JitTensor<R>,
) -> JitTensor<R> {
    let strategy = if tensor.can_mut_broadcast(&value) {
        MaskWhereStrategy::InplaceLhs
    } else if value.can_mut_broadcast(&tensor) {
        MaskWhereStrategy::InplaceRhs
    } else {
        MaskWhereStrategy::Readonly
    };

    super::mask_where::<R, E>(tensor, mask, value, strategy)
}
