use std::marker::PhantomData;

use burn_cube::prelude::*;

use crate::kernel::matmul::tiling2d_cube::load_shared_memory::{
    LoadInfo, LoadInfoExpand, SharedMemoryLoader,
};

use super::loader::Loader;

// Transposed tensor's vectorization must be 1
// On nvidia, version lhs transposed rhs plain:
// cpa version: 250ms
// tile_loader: 263ms
// this one: 235ms
// Warning: does not check bounds
pub(crate) struct DirectLoader<F: Float, L: Loader<F>, R: Loader<F>> {
    _f: PhantomData<F>,
    _lhs: PhantomData<L>,
    _rhs: PhantomData<R>,
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

impl<F: Float, L: Loader<F>, R: Loader<F>> SharedMemoryLoader<F> for DirectLoader<F, L, R> {
    fn load_lhs_plain_expand(
        context: &mut CubeContext,
        lhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    ) {
        load_lhs_plain_expand::<F, L>(context, lhs, load_info);
    }

    fn load_lhs_transposed_expand(
        context: &mut CubeContext,
        lhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    ) {
        load_lhs_transposed_expand::<F, L>(context, lhs, load_info);
    }

    fn load_rhs_plain_expand(
        context: &mut CubeContext,
        rhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    ) {
        load_rhs_plain_expand::<F, R>(context, rhs, load_info);
    }

    fn load_rhs_transposed_expand(
        context: &mut CubeContext,
        rhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    ) {
        load_rhs_transposed_expand::<F, R>(context, rhs, load_info);
    }
}

#[cube]
fn load_lhs_plain<F: Float, L: Loader<F>>(lhs: Tensor<F>, load_info: LoadInfo<F>) {
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

    L::load_plain(lhs, load_info, load_indices, check_bounds);
}

#[cube]
fn load_lhs_transposed<F: Float, L: Loader<F>>(lhs: Tensor<F>, load_info: LoadInfo<F>) {
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

    L::load_transposed(lhs, load_info, load_indices, check_bounds);
}

#[cube]
fn load_rhs_plain<F: Float, L: Loader<F>>(rhs: Tensor<F>, load_info: LoadInfo<F>) {
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

    L::load_plain(rhs, load_info, load_indices, check_bounds);
}

#[cube]
fn load_rhs_transposed<F: Float, L: Loader<F>>(rhs: Tensor<F>, load_info: LoadInfo<F>) {
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

    L::load_transposed(rhs, load_info, load_indices, check_bounds);
}
