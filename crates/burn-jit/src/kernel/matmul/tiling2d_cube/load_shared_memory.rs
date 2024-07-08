use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::{
    base::{BatchOffsets, Coordinates, Dimensions, SharedMemories},
    direct::block_check::{
        base::BlockCheck, horizontal_block_check::HorizontalBlockCheck,
        unchecked_block::UncheckedBlockCheck, vertical_block_check::VerticalBlockCheck,
        whole_block_check::WholeBlockCheck,
    },
};

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
pub(crate) trait Loader<F: Float>: Sync + Send + 'static {
    fn load_lhs_plain<B: BlockCheck<F>>(lhs: &Tensor<F>, load_info: LoadInfo<F>);
    fn load_lhs_transposed<B: BlockCheck<F>>(lhs: &Tensor<F>, load_info: LoadInfo<F>);
    fn load_rhs_plain<B: BlockCheck<F>>(rhs: &Tensor<F>, load_info: LoadInfo<F>);
    fn load_rhs_transposed<B: BlockCheck<F>>(rhs: &Tensor<F>, load_info: LoadInfo<F>);
}

#[cube]
pub(crate) fn load_to_shared_memories<F: Float, L: Loader<F>>(
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

    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);

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
        if Comptime::get(check_k_bounds) {
            if Comptime::get(check_m_bounds) {
                L::load_lhs_plain::<WholeBlockCheck>(lhs, lhs_load_info);
            } else {
                L::load_lhs_plain::<VerticalBlockCheck>(lhs, lhs_load_info);
            }
        } else {
            if Comptime::get(check_m_bounds) {
                L::load_lhs_plain::<HorizontalBlockCheck>(lhs, lhs_load_info);
            } else {
                L::load_lhs_plain::<UncheckedBlockCheck>(lhs, lhs_load_info);
            }
        }
    } else {
        if Comptime::get(check_m_bounds) {
            if Comptime::get(check_k_bounds) {
                L::load_lhs_transposed::<WholeBlockCheck>(lhs, lhs_load_info);
            } else {
                L::load_lhs_transposed::<VerticalBlockCheck>(lhs, lhs_load_info);
            }
        } else {
            if Comptime::get(check_k_bounds) {
                L::load_lhs_transposed::<HorizontalBlockCheck>(lhs, lhs_load_info);
            } else {
                L::load_lhs_transposed::<UncheckedBlockCheck>(lhs, lhs_load_info);
            }
        }
    }

    // Rhs must be loaded as plain. If it is transposed in global memory, we transpose it back.
    if Comptime::get(rhs_transposed) {
        if Comptime::get(check_n_bounds) {
            if Comptime::get(check_k_bounds) {
                L::load_rhs_transposed::<WholeBlockCheck>(rhs, rhs_load_info);
            } else {
                L::load_rhs_transposed::<VerticalBlockCheck>(rhs, rhs_load_info);
            }
        } else {
            if Comptime::get(check_k_bounds) {
                L::load_rhs_transposed::<HorizontalBlockCheck>(rhs, rhs_load_info);
            } else {
                L::load_rhs_transposed::<UncheckedBlockCheck>(rhs, rhs_load_info);
            }
        }
    } else {
        if Comptime::get(check_k_bounds) {
            if Comptime::get(check_n_bounds) {
                L::load_rhs_plain::<WholeBlockCheck>(rhs, rhs_load_info);
            } else {
                L::load_rhs_plain::<VerticalBlockCheck>(rhs, rhs_load_info);
            }
        } else {
            if Comptime::get(check_n_bounds) {
                L::load_rhs_plain::<HorizontalBlockCheck>(rhs, rhs_load_info);
            } else {
                L::load_rhs_plain::<UncheckedBlockCheck>(rhs, rhs_load_info);
            }
        }
    }
}
