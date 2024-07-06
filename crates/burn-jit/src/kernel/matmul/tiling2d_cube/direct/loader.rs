use burn_cube::{prelude::*, unexpanded};

use crate::kernel::matmul::{
    config::CubeTiling2dConfig,
    tiling2d_cube::load_shared_memory::{LoadInfo, LoadInfoExpand},
};

use super::{
    base::{CheckBounds, CheckBoundsExpand, LoadIndices, LoadIndicesExpand},
    plain::{
        horizontal_check_load_plain_expand, load_plain_expand, unchecked_load_plain_expand,
        vertical_check_load_plain_expand, wholly_check_load_plain_expand,
    },
    transpose::{
        horizontal_check_load_transposed_expand, load_transposed_expand,
        unchecked_load_transposed_expand, vertical_check_load_transposed_expand,
        wholly_check_load_transposed_expand,
    },
};

#[derive(CubeType)]
pub(crate) struct ReadTileInfo {
    pub read_row: UInt,
    pub read_col: UInt,
    pub gm_position_base: UInt,
    pub sm_position_base: UInt,
    pub gm_stride: UInt,
    pub sm_stride: UInt,
}

pub(crate) trait Loader<F: Float>: Send + Sync + 'static {
    fn load_transposed(
        _tensor: Tensor<F>,
        _load_info: LoadInfo<F>,
        _load_indices: LoadIndices,
        _check_bounds: CheckBounds,
    ) {
        unexpanded!()
    }

    fn load_transposed_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    );

    fn load_plain(
        _tensor: Tensor<F>,
        _load_info: LoadInfo<F>,
        _load_indices: LoadIndices,
        _check_bounds: CheckBounds,
    ) {
        unexpanded!()
    }

    fn load_plain_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    );

    fn load_tile_transposed(
        _tensor: Tensor<F>,
        _shared_memory: SharedMemory<F>,
        _read_tile_info: ReadTileInfo,
        _config: Comptime<CubeTiling2dConfig>,
        _check_bounds: CheckBounds,
    ) {
        unexpanded!()
    }

    fn load_tile_transposed_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    );

    fn load_tile_plain(
        _tensor: Tensor<F>,
        _shared_memory: SharedMemory<F>,
        _read_tile_info: ReadTileInfo,
        _config: Comptime<CubeTiling2dConfig>,
        _check_bounds: CheckBounds,
    ) {
        unexpanded!()
    }

    fn load_tile_plain_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    );
}

pub(crate) struct UncheckedLoad;
pub(crate) struct VerticallyCheckedLoad;
pub(crate) struct HorizontallyCheckedLoad;
pub(crate) struct WhollyCheckedLoad;

impl<F: Float> Loader<F> for UncheckedLoad {
    fn load_transposed_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_transposed_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds);
    }

    fn load_tile_transposed_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        _check_bounds: CheckBoundsExpand,
    ) {
        unchecked_load_transposed_expand(context, tensor, shared_memory, read_tile_info, config);
    }

    fn load_plain_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_plain_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds)
    }

    fn load_tile_plain_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        _check_bounds: CheckBoundsExpand,
    ) {
        unchecked_load_plain_expand(context, tensor, shared_memory, read_tile_info, config);
    }
}

impl<F: Float> Loader<F> for VerticallyCheckedLoad {
    fn load_transposed_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_transposed_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds);
    }

    fn load_tile_transposed_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    ) {
        vertical_check_load_transposed_expand(
            context,
            tensor,
            shared_memory,
            read_tile_info,
            config,
            check_bounds,
        );
    }

    fn load_plain_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_plain_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds)
    }

    fn load_tile_plain_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    ) {
        vertical_check_load_plain_expand(
            context,
            tensor,
            shared_memory,
            read_tile_info,
            config,
            check_bounds,
        );
    }
}

impl<F: Float> Loader<F> for HorizontallyCheckedLoad {
    fn load_transposed_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_transposed_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds);
    }

    fn load_tile_transposed_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    ) {
        horizontal_check_load_transposed_expand(
            context,
            tensor,
            shared_memory,
            read_tile_info,
            config,
            check_bounds,
        );
    }

    fn load_plain_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_plain_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds)
    }

    fn load_tile_plain_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    ) {
        horizontal_check_load_plain_expand(
            context,
            tensor,
            shared_memory,
            read_tile_info,
            config,
            check_bounds,
        );
    }
}

impl<F: Float> Loader<F> for WhollyCheckedLoad {
    fn load_transposed_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_transposed_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds);
    }

    fn load_tile_transposed_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    ) {
        wholly_check_load_transposed_expand(
            context,
            tensor,
            shared_memory,
            read_tile_info,
            config,
            check_bounds,
        );
    }

    fn load_plain_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_plain_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds)
    }

    fn load_tile_plain_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    ) {
        wholly_check_load_plain_expand(
            context,
            tensor,
            shared_memory,
            read_tile_info,
            config,
            check_bounds,
        );
    }
}
