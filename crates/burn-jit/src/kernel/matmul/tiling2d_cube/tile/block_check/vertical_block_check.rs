use burn_cube::prelude::*;

use crate::kernel::matmul::{
    config::CubeTiling2dConfig,
    tiling2d_cube::{
        tile::{
            loader::{CheckBounds, ReadTileInfo},
            memory_access::{
                ContiguousAccess, StridedAccess, UnmatchingVectorization, WritePositions,
                WritePositionsExpand,
            },
        },
        write_output::WriteTileInfo,
    },
};

use super::base::{all_zeros_runtime, all_zeros_runtime_expand, BlockCheck};

pub(crate) struct VerticalBlockCheck;

#[cube]
impl<F: Float> BlockCheck<F> for VerticalBlockCheck {
    fn load_tile_plain<A: ContiguousAccess<F>>(
        tensor: &Tensor<F>,
        shared_memory: &mut SharedMemory<F>,
        info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
        check_bounds: CheckBounds,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let vectorization = Comptime::vectorization(&tensor);

        let mut num_reads = UInt::new(0);
        let row = check_bounds.skip_row + info.read_row;
        if check_bounds.dim_vertical > row {
            num_reads = UInt::min(
                check_bounds.dim_vertical - row,
                Comptime::runtime(tile_size),
            );
        }

        for i in range(0u32, num_reads, Comptime::new(false)) {
            let gm_position =
                (info.gm_position_base + i * info.gm_stride) / Comptime::runtime(vectorization);
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = A::read_contiguous_unchecked(tensor, gm_position, config);
        }

        all_zeros_runtime(
            shared_memory,
            num_reads,
            info.sm_position_base,
            info.sm_stride,
            config,
        );
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

        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let gm_position = info.gm_position_base + i;
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = UnmatchingVectorization::read_strided_checked(
                tensor,
                gm_position,
                info.gm_stride,
                check_bounds,
                info,
                config,
            );
        }
    }

    fn write_output<A: ContiguousAccess<F>>(
        out: &mut Tensor<F>,
        results: &Array<F>,
        info: WriteTileInfo,
        config: Comptime<CubeTiling2dConfig>,
        check_bounds: CheckBounds,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let coordinates = info.coordinates;

        let row = coordinates.skip_row + coordinates.unit_row;
        let col = coordinates.skip_col + coordinates.unit_col;
        let out_position_base = row * info.out_stride + col + info.offset_output;

        let mut num_writes = UInt::new(0);
        if check_bounds.dim_vertical > row {
            num_writes = UInt::min(
                check_bounds.dim_vertical - row,
                Comptime::runtime(tile_size),
            );
        }

        for result_index in range(0u32, num_writes, Comptime::new(false)) {
            let positions = WritePositions {
                result: result_index * Comptime::runtime(tile_size),
                out: out_position_base + result_index * info.out_stride,
            };

            A::write_contiguous_unchecked(out, results, positions, config);
        }
    }
}
