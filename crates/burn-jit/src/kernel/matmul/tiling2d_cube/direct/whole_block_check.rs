use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::base::{CheckBounds, Loader, ReadTileInfo};

pub(crate) struct WholeBlockCheckLoad;

#[cube]
impl<F: Float> Loader<F> for WholeBlockCheckLoad {
    fn load_tile_plain(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<F>,
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
    fn load_tile_transposed(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<F>,
        info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
        check_bounds: CheckBounds,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);

        let mut num_reads_horizontal = UInt::new(0);
        let col = check_bounds.skip_col + info.read_col;
        let dim_horizontal = check_bounds.dim_horizontal;
        if dim_horizontal > col {
            num_reads_horizontal = UInt::min(dim_horizontal - col, Comptime::runtime(tile_size));
        }

        let mut num_reads_vertical = UInt::new(0);
        let row = check_bounds.skip_row + info.read_row;
        let dim_vertical = check_bounds.dim_vertical;
        if dim_vertical > row {
            num_reads_vertical = UInt::min(dim_vertical - row, Comptime::runtime(tile_size));
        }

        for i in range(0u32, num_reads_horizontal, Comptime::new(false)) {
            let gm_position = info.gm_position_base + i;
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            let mut transposed = F::vectorized_empty(Comptime::get(tile_size));
            for j in range(0u32, num_reads_vertical, Comptime::new(false)) {
                transposed[j] = tensor[gm_position + j * info.gm_stride];
            }
            for j in range(
                num_reads_vertical,
                Comptime::get(tile_size),
                Comptime::new(false),
            ) {
                transposed[j] = F::new(0.);
            }

            shared_memory[sm_position] = transposed;
        }
    }
}
