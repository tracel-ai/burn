use std::marker::PhantomData;

use burn_cube::prelude::*;

use crate::kernel::matmul::{
    config::CubeTiling2dConfig,
    tiling2d_cube::load_shared_memory::{LoadInfo, SharedMemoryLoader},
};

// Transposed tensor's vectorization must be 1
// Plain tensor's vectorization must equal tile size
pub(crate) struct DirectLoader<F: Float, L: Loader<F>, R: Loader<F>> {
    _f: PhantomData<F>,
    _lhs: PhantomData<L>,
    _rhs: PhantomData<R>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BlockCheck {
    Whole,
    Horizontal,
    Vertical,
    Unchecked,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum VectorReaderEnum {
    Matching,
    Unmatching,
}

#[derive(CubeType)]
pub(crate) struct LoadIndices {
    pub offset: UInt,
    pub gm_stride: UInt,
    pub sm_stride: UInt,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct CheckBounds {
    pub dim_vertical: UInt,
    pub dim_horizontal: UInt,
    pub skip_row: UInt,
    pub skip_col: UInt,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct ReadTileInfo {
    pub read_row: UInt,
    pub read_col: UInt,
    pub gm_position_base: UInt,
    pub sm_position_base: UInt,
    pub gm_stride: UInt,
    pub sm_stride: UInt,
}

#[cube]
pub(crate) trait Loader<F: Float>: Send + Sync + 'static {
    fn load_tile_plain(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<F>,
        read_tile_info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
        check_bounds: CheckBounds,
    );
    fn load_tile_transposed(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<F>,
        read_tile_info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
        check_bounds: CheckBounds,
    );
}

#[cube]
impl<F: Float, L: Loader<F>, R: Loader<F>> SharedMemoryLoader<F> for DirectLoader<F, L, R> {
    fn load_lhs_plain(lhs: &Tensor<F>, load_info: LoadInfo<F>) {
        let config = load_info.config;
        let dims = load_info.dims;
        let coordinates = load_info.coordinates;
        let gm_stride = dims.m;

        let load_indices = LoadIndices {
            offset: coordinates.skip_row + load_info.k * gm_stride + load_info.batch_offset,
            gm_stride,
            sm_stride: Comptime::runtime(Comptime::map(config, |c| c.block_size_n)),
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.k,
            dim_horizontal: dims.m,
            skip_row: load_info.k,
            skip_col: coordinates.skip_row,
        };

        load_plain::<F, L>(lhs, load_info, load_indices, check_bounds);
    }

    fn load_lhs_transposed(lhs: &Tensor<F>, load_info: LoadInfo<F>) {
        let config = load_info.config;
        let dims = load_info.dims;
        let coordinates = load_info.coordinates;
        let gm_stride = dims.k;

        let load_indices = LoadIndices {
            offset: coordinates.skip_row * gm_stride + load_info.k + load_info.batch_offset,
            gm_stride,
            sm_stride: Comptime::runtime(Comptime::map(config, |c| c.block_size_m)),
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.m,
            dim_horizontal: dims.k,
            skip_row: coordinates.skip_row,
            skip_col: load_info.k,
        };

        load_transposed::<F, L>(lhs, load_info, load_indices, check_bounds);
    }

    fn load_rhs_plain(rhs: &Tensor<F>, load_info: LoadInfo<F>) {
        let coordinates = load_info.coordinates;
        let dims = load_info.dims;
        let config = load_info.config;
        let gm_stride = dims.n;

        let load_indices = LoadIndices {
            offset: coordinates.skip_col + load_info.k * gm_stride + load_info.batch_offset,
            gm_stride,
            sm_stride: Comptime::runtime(Comptime::map(config, |c| c.block_size_n)),
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.k,
            dim_horizontal: dims.n,
            skip_row: load_info.k,
            skip_col: coordinates.skip_col,
        };

        load_plain::<F, R>(rhs, load_info, load_indices, check_bounds);
    }

    fn load_rhs_transposed(rhs: &Tensor<F>, load_info: LoadInfo<F>) {
        let config = load_info.config;
        let dims = load_info.dims;
        let coordinates = load_info.coordinates;
        let gm_stride = dims.k;

        let load_indices = LoadIndices {
            offset: coordinates.skip_col * gm_stride + load_info.k + load_info.batch_offset,
            gm_stride,
            sm_stride: Comptime::runtime(Comptime::map(config, |c| c.block_size_n)),
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.n,
            dim_horizontal: dims.k,
            skip_row: coordinates.skip_col,
            skip_col: load_info.k,
        };

        load_transposed::<F, R>(rhs, load_info, load_indices, check_bounds);
    }
}

#[cube]
pub(crate) fn load_plain<F: Float, L: Loader<F>>(
    tensor: &Tensor<F>,
    load_info: LoadInfo<F>,
    load_indices: LoadIndices,
    check_bounds: CheckBounds,
) {
    let coordinates = load_info.coordinates;
    let config = load_info.config;

    let sm_dim_vertical = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));

    let read_row = coordinates.unit_row;
    let read_col = coordinates.unit_col;
    let write_row = coordinates.unit_row;
    let write_col = coordinates.unit_col;

    let gm_position_base = read_row * load_indices.gm_stride + read_col + load_indices.offset;
    let sm_position_base = write_row * load_indices.sm_stride + write_col;

    let read_tile_info = ReadTileInfo {
        read_row,
        read_col,
        gm_position_base,
        sm_position_base,
        gm_stride: load_indices.gm_stride,
        sm_stride: load_indices.sm_stride,
    };
    let mut sm = load_info.shared_memory;

    if write_row < sm_dim_vertical {
        L::load_tile_plain(tensor, &mut sm, read_tile_info, config, check_bounds);
    }
}

#[cube]
pub(crate) fn load_transposed<F: Float, L: Loader<F>>(
    tensor: &Tensor<F>,
    load_info: LoadInfo<F>,
    load_indices: LoadIndices,
    check_bounds: CheckBounds,
) {
    let coordinates = load_info.coordinates;
    let config = load_info.config;

    let sm_dim_vertical = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));

    let read_row = coordinates.unit_row;
    let read_col = coordinates.unit_col;
    let write_row = coordinates.unit_col;
    let write_col = coordinates.unit_row;

    let gm_position_base = read_row * load_indices.gm_stride + read_col + load_indices.offset;
    let sm_position_base = write_row * load_indices.sm_stride + write_col;

    let read_tile_info = ReadTileInfo {
        read_row,
        read_col,
        gm_position_base,
        sm_position_base,
        gm_stride: load_indices.gm_stride,
        sm_stride: load_indices.sm_stride,
    };
    let mut sm = load_info.shared_memory;

    if write_row < sm_dim_vertical {
        L::load_tile_transposed(tensor, &mut sm, read_tile_info, config, check_bounds);
    }
}
#[cube]
pub(crate) fn all_zeros_runtime<F: Float>(
    shared_memory: &mut SharedMemory<F>,
    start: UInt,
    sm_position_base: UInt,
    sm_stride: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let zeros = F::vectorized(0., Comptime::get(tile_size));

    for i in range(start, Comptime::get(tile_size), Comptime::new(false)) {
        let sm_position = (sm_position_base + i * sm_stride) / Comptime::runtime(tile_size);

        shared_memory[sm_position] = zeros;
    }
}

#[cube]
pub(crate) fn all_zeros_comptime<F: Float>(
    shared_memory: &mut SharedMemory<F>,
    sm_position_base: UInt,
    sm_stride: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll_tile);
    let zeros = F::vectorized(0., Comptime::get(tile_size));

    for i in range(0u32, Comptime::get(tile_size), unroll) {
        let sm_position = (sm_position_base + i * sm_stride) / Comptime::runtime(tile_size);

        shared_memory[sm_position] = zeros;
    }
}
