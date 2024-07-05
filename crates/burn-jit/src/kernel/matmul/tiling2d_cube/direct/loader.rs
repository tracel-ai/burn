use burn_cube::prelude::*;

use crate::kernel::matmul::tiling2d_cube::{
    load_shared_memory::{LoadInfo, LoadInfoExpand, SharedMemoryLoader},
    tile,
};

// LHS vectorization must be 1
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
fn load_lhs_plain<F: Float>(lhs: Tensor<F>, load_info: LoadInfo<F>) {}

#[cube]
fn load_lhs_transposed<F: Float>(lhs: Tensor<F>, load_info: LoadInfo<F>) {
    let config = load_info.config;
    let dims = load_info.dims;
    let coordinates = load_info.coordinates;
    let k = load_info.k;
    let batch_offset = load_info.batch_offset;
    let mut shared_memory = load_info.shared;

    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let tile_size = Comptime::map(config, |c| c.tile_size);

    let sm_stride = Comptime::runtime(block_size_m);
    let unroll = Comptime::map(config, |c| c.unroll_tile);
    let sm_dim_vertical = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));

    let tensor_stride = dims.k;
    let offset = coordinates.skip_row * tensor_stride + k + batch_offset;

    // Would be interesting to check the reverse, so that SM write is more coalesced than GM read
    let read_row = coordinates.unit_row;
    let read_col = coordinates.unit_col;
    let write_row = read_col;
    let write_col = read_row;

    let tensor_position_base = read_row * tensor_stride + read_col + offset;
    let sm_position_base = write_row * sm_stride + write_col;

    if write_row < sm_dim_vertical {
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let gm_position = tensor_position_base + i;
            let sm_position = (sm_position_base + i * sm_stride) / Comptime::runtime(tile_size);

            let mut transposed = F::vectorized_empty(Comptime::get(tile_size));
            for j in range(0u32, Comptime::get(tile_size), unroll) {
                transposed[j] = lhs[gm_position + j * tensor_stride];
            }

            shared_memory[sm_position] = transposed;
        }
    }
}

#[cube]
fn load_rhs_plain<F: Float>(rhs: Tensor<F>, load_info: LoadInfo<F>) {
    let config = load_info.config;
    let dims = load_info.dims;
    let coordinates = load_info.coordinates;
    let k = load_info.k;
    let batch_offset = load_info.batch_offset;
    let mut shared_memory = load_info.shared;

    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let runtime_tile_size = Comptime::runtime(tile_size);

    let sm_stride = Comptime::runtime(block_size_m);
    let unroll = Comptime::map(config, |c| c.unroll_tile);
    let sm_dim_vertical = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
    let vectorization = Comptime::vectorization(&rhs);
    let runtime_vectorization = Comptime::runtime(vectorization);

    let read_row = coordinates.unit_row;
    let read_col = coordinates.unit_col;
    let write_row = read_row;
    let write_col = read_col;

    let tensor_stride = dims.n;
    let offset = coordinates.skip_col + k * tensor_stride + batch_offset;
    let tensor_position_base = read_row * tensor_stride + read_col + offset;
    let sm_position_base = write_row * sm_stride + write_col;

    if write_row < sm_dim_vertical {
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let tensor_position = tensor_position_base + i * tensor_stride;
            let sm_position = (sm_position_base + i * sm_stride) / runtime_tile_size;

            shared_memory[sm_position] = rhs[tensor_position / runtime_vectorization];
        }
    }
}

#[cube]
fn load_rhs_transposed<F: Float>(rhs: Tensor<F>, load_info: LoadInfo<F>) {}
