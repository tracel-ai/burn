use burn_backend::DType;
use cubecl::prelude::InputScalar;

use super::{MaskFillStrategy, mask_where::MaskWhereStrategy};
use crate::tensor::CubeTensor;

/// Execute the mask fill kernel.
pub(crate) fn mask_fill_auto(
    tensor: CubeTensor,
    mask: CubeTensor,
    value: InputScalar,
    dtype_bool: DType,
) -> CubeTensor {
    let strategy = if tensor.can_mut() && tensor.is_nonoverlapping() {
        MaskFillStrategy::Inplace
    } else {
        MaskFillStrategy::Readonly
    };

    super::mask_fill(tensor, mask, value, strategy, dtype_bool)
}

/// Execute the mask where kernel.
pub(crate) fn mask_where_auto(
    tensor: CubeTensor,
    mask: CubeTensor,
    value: CubeTensor,
    dtype_bool: DType,
) -> CubeTensor {
    let strategy = if tensor.can_mut_broadcast(&value) {
        MaskWhereStrategy::InplaceLhs
    } else if value.can_mut_broadcast(&tensor) {
        MaskWhereStrategy::InplaceRhs
    } else {
        MaskWhereStrategy::Readonly
    };

    super::mask_where(tensor, mask, value, strategy, dtype_bool)
}
