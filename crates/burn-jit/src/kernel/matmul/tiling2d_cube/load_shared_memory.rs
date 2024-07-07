use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::base::{BatchOffsets, Coordinates, Dimensions, SharedMemories};

#[derive(CubeType)]
#[allow(dead_code)]
pub(crate) struct LoadInfo<F: Float> {
    pub coordinates: Coordinates,
    pub k: UInt,
    pub batch_offset: UInt,
    pub shared_memory: SharedMemory<F>,
    pub config: Comptime<CubeTiling2dConfig>,
    pub dims: Dimensions,
}

#[cube]
pub(crate) trait SharedMemoryLoader<F: Float>: Sync + Send + 'static {
    fn load_lhs_plain(lhs: &Tensor<F>, load_info: LoadInfo<F>);
    fn load_lhs_transposed(lhs: &Tensor<F>, load_info: LoadInfo<F>);
    fn load_rhs_plain(rhs: &Tensor<F>, load_info: LoadInfo<F>);
    fn load_rhs_transposed(rhs: &Tensor<F>, load_info: LoadInfo<F>);
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
        shared_memory: shared.lhs,
        config,
        dims,
    };
    let rhs_load_info = LoadInfo {
        coordinates,
        k,
        batch_offset: offsets.rhs,
        shared_memory: shared.rhs,
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
