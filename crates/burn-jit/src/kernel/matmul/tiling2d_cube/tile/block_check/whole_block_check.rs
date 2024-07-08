use burn_cube::prelude::*;

use crate::kernel::matmul::{
    config::CubeTiling2dConfig,
    tiling2d_cube::{
        base::Coordinates,
        tile::{
            loader::{CheckBounds, ReadTileInfo},
            memory_access::{ContiguousAccess, StridedAccess, UnmatchingVectorization},
        },
    },
};

use super::base::{
    all_zeros_comptime, all_zeros_comptime_expand, all_zeros_runtime, all_zeros_runtime_expand,
    BlockCheck,
};

pub(crate) struct WholeBlockCheck;

#[cube]
impl<F: Float> BlockCheck<F> for WholeBlockCheck {
    fn load_tile_plain<A: ContiguousAccess<F>>(
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

                shared_memory[sm_position] =
                    A::read_contiguous_checked(tensor, gm_position, check_bounds, info, config);
            }

            all_zeros_runtime(
                shared_memory,
                num_reads_vertical,
                info.sm_position_base,
                info.sm_stride,
                config,
            );
        } else {
            all_zeros_comptime(shared_memory, info.sm_position_base, info.sm_stride, config);
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

        for i in range(0u32, num_reads_horizontal, Comptime::new(false)) {
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

        all_zeros_runtime(
            shared_memory,
            num_reads_horizontal,
            info.sm_position_base,
            info.sm_stride,
            config,
        );
    }

    fn write_output<A: ContiguousAccess<F>>(
        out: &mut Tensor<F>,
        results: &Array<F>,
        coordinates: Coordinates,
        offset_output: UInt,
        out_stride: UInt,
        config: Comptime<CubeTiling2dConfig>,
        check_bounds: CheckBounds,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);

        let col = coordinates.skip_col + coordinates.unit_col;

        if check_bounds.dim_horizontal > col {
            let mut num_writes_vertical = UInt::new(0);
            let row = coordinates.skip_row + coordinates.unit_row;

            if check_bounds.dim_vertical > row {
                num_writes_vertical = UInt::min(
                    check_bounds.dim_vertical - row,
                    Comptime::runtime(tile_size),
                );
            }

            let out_position_base = row * out_stride + col + offset_output;

            for result_index in range(0u32, num_writes_vertical, Comptime::new(false)) {
                let result_position = result_index * Comptime::runtime(tile_size);
                let out_position = out_position_base + result_index * out_stride;

                A::write_contiguous_checked(
                    out,
                    out_position,
                    results,
                    result_position,
                    check_bounds,
                    col,
                    config,
                );
            }
        }
    }
}
