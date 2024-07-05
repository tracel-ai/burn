use std::marker::PhantomData;

use burn_cube::prelude::*;

use crate::kernel::matmul::{
    config::CubeTiling2dConfig,
    tiling2d_cube::{
        base::Coordinates,
        load_shared_memory::{LoadInfo, LoadInfoExpand, SharedMemoryLoader},
    },
};

use super::transpose_trait::TransposeLoad;

// Transposed tensor's vectorization must be 1
// On nvidia, version lhs transposed rhs plain:
// cpa version: 250ms
// tile_loader: 263ms
// this one: 235ms
// Warning: does not check bounds
pub(crate) struct DirectLoader<F: Float, T: TransposeLoad<F>> {
    _f: PhantomData<F>,
    _t: PhantomData<T>,
}

#[derive(CubeType)]
pub(crate) struct LoadIndices {
    pub offset: UInt,
    pub gm_stride: UInt,
    pub sm_stride: UInt,
}

#[derive(CubeType)]
pub(crate) struct CheckBounds {
    pub dim_vertical: UInt,
    pub dim_horizontal: UInt,
    pub skip_row: UInt,
    pub skip_col: UInt,
}

impl<F: Float, T: TransposeLoad<F>> SharedMemoryLoader<F> for DirectLoader<F, T> {
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
        load_lhs_transposed_expand::<F, T>(context, lhs, load_info);
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
        load_rhs_transposed_expand::<F, T>(context, rhs, load_info);
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
    // let check_bounds = CheckBounds {
    //     dim_vertical: dims.m,
    //     dim_horizontal: dims.k,
    //     skip_row: load_info.k,
    //     skip_col: coordinates.skip_row,
    // };

    load_plain(&lhs, load_info.shared, load_indices, coordinates, config);
}

#[cube]
fn load_lhs_transposed<F: Float, T: TransposeLoad<F>>(lhs: Tensor<F>, load_info: LoadInfo<F>) {
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

    T::load(lhs, load_info, load_indices, check_bounds);
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
    // let check_bounds = CheckBounds {
    //     dim_vertical: dims.k,
    //     dim_horizontal: dims.n,
    //     skip_row: load_info.k,
    //     skip_col: coordinates.skip_col,
    // };

    load_plain(&rhs, load_info.shared, load_indices, coordinates, config);
}

#[cube]
fn load_rhs_transposed<F: Float, T: TransposeLoad<F>>(rhs: Tensor<F>, load_info: LoadInfo<F>) {
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
        dim_vertical: dims.k,
        dim_horizontal: dims.n,
        skip_row: coordinates.skip_col,
        skip_col: load_info.k,
    };

    T::load(rhs, load_info, load_indices, check_bounds);
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
