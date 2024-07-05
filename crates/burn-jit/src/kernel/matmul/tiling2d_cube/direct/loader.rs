use burn_cube::prelude::*;

use crate::kernel::matmul::{
    config::CubeTiling2dConfig,
    tiling2d_cube::{
        base::Coordinates,
        load_shared_memory::{LoadInfo, LoadInfoExpand, SharedMemoryLoader},
    },
};

// Transposed tensor's vectorization must be 1
// On nvidia:
// cpa version: 250ms
// tile_loader: 263ms
// this one: 235ms
// but does not check bounds yet
pub(crate) struct DirectLoader;

impl<F: Float> SharedMemoryLoader<F> for DirectLoader {
    fn load_lhs_plain_expand(
        context: &mut CubeContext,
        lhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    ) {
        load_lhs_plain_expand(context, lhs, load_info);
    }

    fn load_lhs_transposed_expand(
        context: &mut CubeContext,
        lhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    ) {
        load_lhs_transposed_expand(context, lhs, load_info);
    }

    fn load_rhs_plain_expand(
        context: &mut CubeContext,
        rhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    ) {
        load_rhs_plain_expand(context, rhs, load_info);
    }

    fn load_rhs_transposed_expand(
        context: &mut CubeContext,
        rhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    ) {
        load_rhs_transposed_expand(context, rhs, load_info);
    }
}

#[cube]
fn load_lhs_plain<F: Float>(lhs: Tensor<F>, load_info: LoadInfo<F>) {
    let config = load_info.config;
    let dims = load_info.dims;
    let coordinates = load_info.coordinates;
    let gm_stride = dims.m;

    let load_indices = LoadIndices {
        offset: coordinates.skip_row + load_info.k * gm_stride + load_info.batch_offset,
        gm_stride,
        sm_stride: Comptime::runtime(Comptime::map(config, |c| c.block_size_n)),
    };

    load_plain(&lhs, load_info.shared, load_indices, coordinates, config);
}

#[cube]
fn load_lhs_transposed<F: Float>(lhs: Tensor<F>, load_info: LoadInfo<F>) {
    let config = load_info.config;
    let dims = load_info.dims;
    let coordinates = load_info.coordinates;
    let gm_stride = dims.k;

    let load_indices = LoadIndices {
        offset: coordinates.skip_row * gm_stride + load_info.k + load_info.batch_offset,
        gm_stride,
        sm_stride: Comptime::runtime(Comptime::map(config, |c| c.block_size_m)),
    };

    load_transposed(&lhs, load_info.shared, load_indices, coordinates, config);
}

#[cube]
fn load_rhs_plain<F: Float>(rhs: Tensor<F>, load_info: LoadInfo<F>) {
    let coordinates = load_info.coordinates;
    let dims = load_info.dims;
    let config = load_info.config;
    let gm_stride = dims.n;

    let load_indices = LoadIndices {
        offset: coordinates.skip_col + load_info.k * gm_stride + load_info.batch_offset,
        gm_stride,
        sm_stride: Comptime::runtime(Comptime::map(config, |c| c.block_size_n)),
    };

    load_plain(&rhs, load_info.shared, load_indices, coordinates, config);
}

#[cube]
fn load_rhs_transposed<F: Float>(rhs: Tensor<F>, load_info: LoadInfo<F>) {
    let config = load_info.config;
    let dims = load_info.dims;
    let coordinates = load_info.coordinates;
    let gm_stride = dims.k;

    let load_indices = LoadIndices {
        offset: coordinates.skip_col * gm_stride + load_info.k + load_info.batch_offset,
        gm_stride,
        sm_stride: Comptime::runtime(Comptime::map(config, |c| c.block_size_n)),
    };

    load_transposed(&rhs, load_info.shared, load_indices, coordinates, config);
}

#[derive(CubeType)]
struct LoadIndices {
    offset: UInt,
    gm_stride: UInt,
    sm_stride: UInt,
}

#[cube]
fn load_plain<F: Float>(
    tensor: &Tensor<F>,
    mut shared_memory: SharedMemory<F>,
    l: LoadIndices,
    coordinates: Coordinates,
    config: Comptime<CubeTiling2dConfig>,
) {
    let vectorization = Comptime::vectorization(tensor);
    let sm_dim_vertical = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll_tile);

    let read_row = coordinates.unit_row;
    let read_col = coordinates.unit_col;
    let write_row = coordinates.unit_row;
    let write_col = coordinates.unit_col;

    let gm_position_base = read_row * l.gm_stride + read_col + l.offset;
    let sm_position_base = write_row * l.sm_stride + write_col;

    if write_row < sm_dim_vertical {
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let tensor_position = gm_position_base + i * l.gm_stride;
            let sm_position = (sm_position_base + i * l.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = tensor[tensor_position / Comptime::runtime(vectorization)];
        }
    }
}

#[cube]
fn load_transposed<F: Float>(
    tensor: &Tensor<F>,
    mut shared_memory: SharedMemory<F>,
    l: LoadIndices,
    coordinates: Coordinates,
    config: Comptime<CubeTiling2dConfig>,
) {
    let sm_dim_vertical = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll_tile);

    // Would be interesting to check the reverse, so that SM write is more coalesced than GM read
    let read_row = coordinates.unit_row;
    let read_col = coordinates.unit_col;
    let write_row = coordinates.unit_col;
    let write_col = coordinates.unit_row;

    let gm_position_base = read_row * l.gm_stride + read_col + l.offset;
    let sm_position_base = write_row * l.sm_stride + write_col;

    if write_row < sm_dim_vertical {
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let gm_position = gm_position_base + i;
            let sm_position = (sm_position_base + i * l.sm_stride) / Comptime::runtime(tile_size);

            let mut transposed = F::vectorized_empty(Comptime::get(tile_size));
            for j in range(0u32, Comptime::get(tile_size), unroll) {
                transposed[j] = tensor[gm_position + j * l.gm_stride];
            }

            shared_memory[sm_position] = transposed;
        }
    }
}
