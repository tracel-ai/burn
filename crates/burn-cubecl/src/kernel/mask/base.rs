use burn_tensor::DType;
use cubecl::std::scalar::InputScalar;

use super::{MaskFillStrategy, mask_where::MaskWhereStrategy};
use crate::{CubeRuntime, tensor::CubeTensor};

/// Execute the mask fill kernel.
pub(crate) fn mask_fill_auto<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: InputScalar,
    dtype_bool: DType,
) -> CubeTensor<R> {
    let strategy = if tensor.can_mut() {
        MaskFillStrategy::Inplace
    } else {
        MaskFillStrategy::Readonly
    };

    super::mask_fill::<R>(tensor, mask, value, strategy, dtype_bool)
}

/// Execute the mask where kernel.
pub(crate) fn mask_where_auto<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    mask: CubeTensor<R>,
    value: CubeTensor<R>,
    dtype_bool: DType,
) -> CubeTensor<R> {
    let strategy = if tensor.can_mut_broadcast(&value) {
        MaskWhereStrategy::InplaceLhs
    } else if value.can_mut_broadcast(&tensor) {
        MaskWhereStrategy::InplaceRhs
    } else {
        MaskWhereStrategy::Readonly
    };

    super::mask_where::<R>(tensor, mask, value, strategy, dtype_bool)
}
