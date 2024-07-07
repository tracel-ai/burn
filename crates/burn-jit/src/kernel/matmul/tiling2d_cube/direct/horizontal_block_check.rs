use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::base::{CheckBounds, Loader, ReadTileInfo};

pub(crate) struct HorizontalBlockCheckLoad;

#[cube]
impl<F: Float> Loader<F> for HorizontalBlockCheckLoad {
    fn load_tile_plain(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<F>,
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

    fn load_tile_transposed(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<F>,
        info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
        check_bounds: CheckBounds,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let unroll = Comptime::map(config, |c| c.unroll_tile);

        let mut num_reads = UInt::new(0);
        let col = check_bounds.skip_col + info.read_col;
        let dim_horizontal = check_bounds.dim_horizontal;
        if dim_horizontal > col {
            num_reads = UInt::min(dim_horizontal - col, Comptime::runtime(tile_size));
        }

        for i in range(0u32, num_reads, Comptime::new(false)) {
            let gm_position = info.gm_position_base + i;
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            let mut transposed = F::vectorized_empty(Comptime::get(tile_size));
            for j in range(0u32, Comptime::get(tile_size), unroll) {
                transposed[j] = tensor[gm_position + j * info.gm_stride];
            }

            shared_memory[sm_position] = transposed;
        }

        let zeros = F::vectorized(0., Comptime::get(tile_size));
        for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = zeros;
        }
    }
}
