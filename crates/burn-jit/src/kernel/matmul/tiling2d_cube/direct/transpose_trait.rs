use burn_cube::{prelude::*, unexpanded};

use crate::kernel::matmul::{
    config::CubeTiling2dConfig,
    tiling2d_cube::load_shared_memory::{LoadInfo, LoadInfoExpand},
};

use super::{
    loader::{CheckBounds, CheckBoundsExpand, LoadIndices, LoadIndicesExpand},
    transpose::{
        horizontal_check_load_expand, load_transposed_expand, unchecked_load_expand,
        vertical_check_load_expand, wholly_check_load_expand, ReadTileInfo, ReadTileInfoExpand,
    },
};

pub(crate) trait TransposeLoad<F: Float>: Send + Sync + 'static {
    fn load(
        _tensor: Tensor<F>,
        _load_info: LoadInfo<F>,
        _load_indices: LoadIndices,
        _check_bounds: CheckBounds,
    ) {
        unexpanded!()
    }
    fn load_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    );

    fn tile_load(
        _tensor: Tensor<F>,
        _shared_memory: SharedMemory<F>,
        _read_tile_info: ReadTileInfo,
        _config: Comptime<CubeTiling2dConfig>,
        _check_bounds: CheckBounds,
    ) {
        unexpanded!()
    }
    fn tile_load_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    );
}

pub(crate) struct UncheckedTransposeLoad;
pub(crate) struct VerticallyCheckedTransposeLoad;
pub(crate) struct HorizontallyCheckedTransposeLoad;
pub(crate) struct WhollyCheckedTransposeLoad;

impl<F: Float> TransposeLoad<F> for UncheckedTransposeLoad {
    fn load_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_transposed_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds);
    }

    fn tile_load_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        _check_bounds: CheckBoundsExpand,
    ) {
        unchecked_load_expand(context, tensor, shared_memory, read_tile_info, config);
    }
}

impl<F: Float> TransposeLoad<F> for VerticallyCheckedTransposeLoad {
    fn load_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_transposed_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds);
    }

    fn tile_load_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    ) {
        vertical_check_load_expand(
            context,
            tensor,
            shared_memory,
            read_tile_info,
            config,
            check_bounds,
        );
    }
}

impl<F: Float> TransposeLoad<F> for HorizontallyCheckedTransposeLoad {
    fn load_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_transposed_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds);
    }

    fn tile_load_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    ) {
        horizontal_check_load_expand(
            context,
            tensor,
            shared_memory,
            read_tile_info,
            config,
            check_bounds,
        );
    }
}

impl<F: Float> TransposeLoad<F> for WhollyCheckedTransposeLoad {
    fn load_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
        load_indices: LoadIndicesExpand,
        check_bounds: CheckBoundsExpand,
    ) {
        load_transposed_expand::<F, Self>(context, tensor, load_info, load_indices, check_bounds);
    }

    fn tile_load_expand(
        context: &mut CubeContext,
        tensor: <Tensor<F> as CubeType>::ExpandType,
        shared_memory: <SharedMemory<F> as CubeType>::ExpandType,
        read_tile_info: ReadTileInfoExpand,
        config: CubeTiling2dConfig,
        check_bounds: CheckBoundsExpand,
    ) {
        wholly_check_load_expand(
            context,
            tensor,
            shared_memory,
            read_tile_info,
            config,
            check_bounds,
        );
    }
}
