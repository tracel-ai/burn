use cubecl::prelude::*;

/// Index into a pitched output tensor (must be contiguous_pitched)
#[cube]
pub(crate) fn index_pitched<E: CubePrimitive>(
    tensor: &Tensor<Line<E>>,
    offset: u32,
    #[comptime] pitched: bool,
) -> u32 {
    let rank = tensor.rank();
    let mut offset = offset;

    if pitched {
        let offset_abs = offset * tensor.line_size();
        let x = offset_abs % tensor.shape(rank - 1);
        let y = offset_abs / tensor.shape(rank - 1);
        let offset_adj = y * tensor.stride(rank - 2) + x;
        offset = offset_adj / tensor.line_size();
    }

    offset
}
