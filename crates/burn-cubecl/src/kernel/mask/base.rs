use super::{MaskFillStrategy, mask_where::MaskWhereStrategy};
use crate::{BoolElement, CubeRuntime, element::CubeElement, tensor::CubeTensor};

/// Execute the mask fill kernel.
pub(crate) fn mask_fill_auto<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    tensor: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: E,
) -> CubeTensor<R> {
    let strategy = if tensor.can_mut() {
        MaskFillStrategy::Inplace
    } else {
        MaskFillStrategy::Readonly
    };

    super::mask_fill::<R, E, BT>(tensor, mask, value, strategy)
}

/// Execute the mask where kernel.
pub(crate) fn mask_where_auto<R: CubeRuntime, E: CubeElement, BT: BoolElement>(
    tensor: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: CubeTensor<R>,
) -> CubeTensor<R> {
    let strategy = if tensor.can_mut_broadcast(&value) {
        MaskWhereStrategy::InplaceLhs
    } else if value.can_mut_broadcast(&tensor) {
        MaskWhereStrategy::InplaceRhs
    } else {
        MaskWhereStrategy::Readonly
    };

    super::mask_where::<R, E, BT>(tensor, mask, value, strategy)
}
