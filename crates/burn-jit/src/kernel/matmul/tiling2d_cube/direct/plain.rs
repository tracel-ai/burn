use ::burn_cube::prelude::*;

use crate::kernel::matmul::{
    config::CubeTiling2dConfig, tiling2d_cube::load_shared_memory::LoadInfo,
};

use super::{
    base::{CheckBounds, LoadIndices},
    loader::{Loader, ReadTileInfo, ReadTileInfoExpand},
};

#[cube]
pub(crate) fn load_plain<F: Float, L: Loader<F>>(
    tensor: Tensor<F>,
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

    if write_row < sm_dim_vertical {
        L::load_tile_plain(
            tensor,
            load_info.shared,
            read_tile_info,
            config,
            check_bounds,
        );
    }
}

#[cube]
pub(crate) fn unchecked_load_plain<F: Float>(
    tensor: Tensor<F>,
    mut shared_memory: SharedMemory<F>,
    info: ReadTileInfo,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll_tile);
    let vectorization = Comptime::vectorization(&tensor);

    for i in range(0u32, Comptime::get(tile_size), unroll) {
        let tensor_position = info.gm_position_base + i * info.gm_stride;
        let sm_position =
            (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

        shared_memory[sm_position] = tensor[tensor_position / Comptime::runtime(vectorization)];
    }
}

#[cube]
pub(crate) fn vertical_check_load_plain<F: Float>(
    tensor: Tensor<F>,
    mut shared_memory: SharedMemory<F>,
    info: ReadTileInfo,
    config: Comptime<CubeTiling2dConfig>,
    check_bounds: CheckBounds,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let vectorization = Comptime::vectorization(&tensor);

    let mut num_reads = UInt::new(0);
    let row = check_bounds.skip_row + info.read_row;
    if check_bounds.dim_vertical > row {
        num_reads = UInt::min(
            check_bounds.dim_vertical - row,
            Comptime::runtime(tile_size),
        );
    }

    for i in range(0u32, num_reads, Comptime::new(false)) {
        let gm_position =
            (info.gm_position_base + i * info.gm_stride) / Comptime::runtime(vectorization);
        let sm_position =
            (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

        shared_memory[sm_position] = tensor[gm_position];
    }

    let zeros = F::vectorized(0., Comptime::get(tile_size));
    for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
        let sm_position =
            (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

        shared_memory[sm_position] = zeros;
    }
}

#[cube]
pub(crate) fn horizontal_check_load_plain<F: Float>(
    tensor: Tensor<F>,
    mut shared_memory: SharedMemory<F>,
    info: ReadTileInfo,
    config: Comptime<CubeTiling2dConfig>,
    check_bounds: CheckBounds,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let vectorization = Comptime::vectorization(&tensor);
    let unroll = Comptime::map(config, |c| c.unroll_tile);

    let col = check_bounds.skip_col + info.read_col;
    if check_bounds.dim_horizontal > col {
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let gm_position =
                (info.gm_position_base + i * info.gm_stride) / Comptime::runtime(vectorization);
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = tensor[gm_position];
        }
    } else {
        let zeros = F::vectorized(0., Comptime::get(tile_size));
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = zeros;
        }
    }
}

#[cube]
pub(crate) fn wholly_check_load_plain<F: Float>(
    tensor: Tensor<F>,
    mut shared_memory: SharedMemory<F>,
    info: ReadTileInfo,
    config: Comptime<CubeTiling2dConfig>,
    check_bounds: CheckBounds,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let vectorization = Comptime::vectorization(&tensor);

    let col = check_bounds.skip_col + info.read_col;
    if check_bounds.dim_horizontal > col {
        let mut num_reads_vertical = UInt::new(0);
        let row = check_bounds.skip_row + info.read_row;
        if check_bounds.dim_vertical > row {
            num_reads_vertical = UInt::min(
                check_bounds.dim_vertical - row,
                Comptime::runtime(tile_size),
            );
        }

        for i in range(0u32, num_reads_vertical, Comptime::new(false)) {
            let gm_position =
                (info.gm_position_base + i * info.gm_stride) / Comptime::runtime(vectorization);
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = tensor[gm_position];
        }

        let zeros = F::vectorized(0., Comptime::get(tile_size));
        for i in range(
            num_reads_vertical,
            Comptime::get(tile_size),
            Comptime::new(false),
        ) {
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = zeros;
        }
    } else {
        let zeros = F::vectorized(0., Comptime::get(tile_size));
        for i in range(0u32, Comptime::get(tile_size), Comptime::new(false)) {
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = zeros;
        }
    }
}
