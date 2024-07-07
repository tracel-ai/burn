use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::base::{CheckBounds, Loader, ReadTileInfo};

/// Assumes block sizes divide tensor shape
pub(crate) struct UncheckedBlockLoad;

#[cube]
impl<F: Float> Loader<F> for UncheckedBlockLoad {
    fn load_tile_plain(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<F>,
        info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
        _check_bounds: CheckBounds,
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

    fn load_tile_transposed(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<F>,
        info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
        _check_bounds: CheckBounds,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let unroll = Comptime::map(config, |c| c.unroll_tile);

        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let gm_position = info.gm_position_base + i;
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            let mut transposed = F::vectorized_empty(Comptime::get(tile_size));
            for j in range(0u32, Comptime::get(tile_size), unroll) {
                transposed[j] = tensor[gm_position + j * info.gm_stride];
            }

            shared_memory[sm_position] = transposed;
        }
    }
}
