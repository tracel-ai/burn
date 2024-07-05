use burn_cube::{prelude::*, unexpanded};

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::base::{BatchOffsets, Coordinates, Dimensions, SharedMemories};

#[derive(CubeType)]
#[allow(dead_code)]
pub(crate) struct LoadInfo<F: Float> {
    pub coordinates: Coordinates,
    pub k: UInt,
    pub batch_offset: UInt,
    pub shared: SharedMemory<F>,
    pub config: Comptime<CubeTiling2dConfig>,
    pub dims: Dimensions,
}

pub(crate) trait SharedMemoryLoader<F: Float>: Sync + Send + 'static {
    fn load_lhs_plain(_lhs: &Tensor<F>, _load_info: LoadInfo<F>) {
        unexpanded!()
    }
    fn load_lhs_transposed(_lhs: &Tensor<F>, _load_info: LoadInfo<F>) {
        unexpanded!()
    }
    fn load_rhs_plain(_rhs: &Tensor<F>, _load_info: LoadInfo<F>) {
        unexpanded!()
    }
    fn load_rhs_transposed(_rhs: &Tensor<F>, _load_info: LoadInfo<F>) {
        unexpanded!()
    }

    fn load_lhs_plain_expand(
        context: &mut CubeContext,
        lhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    );
    fn load_lhs_transposed_expand(
        context: &mut CubeContext,
        lhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    );
    fn load_rhs_plain_expand(
        context: &mut CubeContext,
        rhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    );
    fn load_rhs_transposed_expand(
        context: &mut CubeContext,
        rhs: <Tensor<F> as CubeType>::ExpandType,
        load_info: LoadInfoExpand<F>,
    );
}

#[cube]
pub(crate) fn load_to_shared_memories<F: Float, S: SharedMemoryLoader<F>>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    coordinates: Coordinates,
    k: UInt,
    offsets: BatchOffsets,
    shared: SharedMemories<F>,
    config: Comptime<CubeTiling2dConfig>,
    dims: Dimensions,
) {
    let lhs_transposed = Comptime::map(config, |c| c.lhs_transposed);
    let rhs_transposed = Comptime::map(config, |c| c.rhs_transposed);

    let lhs_load_info = LoadInfo {
        coordinates,
        k,
        batch_offset: offsets.lhs,
        shared: shared.lhs,
        config,
        dims,
    };
    let rhs_load_info = LoadInfo {
        coordinates,
        k,
        batch_offset: offsets.rhs,
        shared: shared.rhs,
        config,
        dims,
    };

    // Lhs must be loaded as transposed. If it already is transposed in global memory, we load as plain.
    if Comptime::get(lhs_transposed) {
        S::load_lhs_plain(lhs, lhs_load_info);
    } else {
        S::load_lhs_transposed(lhs, lhs_load_info);
    }

    // Rhs must be loaded as plain. If it is transposed in global memory, we transpose it back.
    if Comptime::get(rhs_transposed) {
        S::load_rhs_transposed(rhs, rhs_load_info);
    } else {
        S::load_rhs_plain(rhs, rhs_load_info);
    }
}
