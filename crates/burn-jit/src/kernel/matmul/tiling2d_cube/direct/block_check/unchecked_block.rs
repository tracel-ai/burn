use burn_cube::prelude::*;

use crate::kernel::matmul::{
    config::CubeTiling2dConfig,
    tiling2d_cube::{
        base::Coordinates,
        direct::{
            loader::{CheckBounds, ReadTileInfo},
            memory_access::{ContiguousAccess, StridedAccess, UnmatchingVectorization},
        },
    },
};

use super::base::BlockCheck;

/// Assumes block sizes divide tensor shape
pub(crate) struct UncheckedBlockCheck;

#[cube]
impl<F: Float> BlockCheck<F> for UncheckedBlockCheck {
    fn load_tile_plain<A: ContiguousAccess<F>>(
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
            let gm_position =
                (info.gm_position_base + i * info.gm_stride) / Comptime::runtime(vectorization);
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = A::read_contiguous_unchecked(tensor, gm_position, config);
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

            shared_memory[sm_position] = UnmatchingVectorization::read_strided_unchecked(
                tensor,
                gm_position,
                info.gm_stride,
                config,
            );
        }
    }

    fn write_output<A: ContiguousAccess<F>>(
        out: &mut Tensor<F>,
        results: &Array<F>,
        coordinates: Coordinates,
        offset_output: UInt,
        out_stride: UInt,
        config: Comptime<CubeTiling2dConfig>,
        _check_bounds: CheckBounds,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let unroll = Comptime::map(config, |c| c.unroll_tile);

        let row = coordinates.skip_row + coordinates.unit_row;
        let col = coordinates.skip_col + coordinates.unit_col;
        let out_base_position = row * out_stride + col + offset_output;

        for result_index in range(0u32, Comptime::get(tile_size), unroll) {
            let result_position = result_index * Comptime::runtime(tile_size);
            let out_position = out_base_position + result_index * out_stride;

            A::write_contiguous_unchecked(out, out_position, results, result_position, config);
        }
    }
}
