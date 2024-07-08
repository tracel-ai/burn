use std::marker::PhantomData;

use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::{
    base::{all_zeros_runtime, all_zeros_runtime_expand, CheckBounds, Loader, ReadTileInfo},
    vector_reader::{HorizontalReader, UnmatchingVectorReader, VerticalReader},
};

pub(crate) struct VerticalBlockCheckLoad<H> {
    _h: PhantomData<H>,
}

#[cube]
impl<F: Float, V: HorizontalReader<F>> Loader<F> for VerticalBlockCheckLoad<V> {
    fn load_tile_plain(
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

            shared_memory[sm_position] = V::read_horizontal_unchecked(tensor, gm_position, config);
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

            shared_memory[sm_position] = UnmatchingVectorReader::read_vertical_checked(
                tensor,
                gm_position,
                info.gm_stride,
                check_bounds,
                info,
                config,
            );
        }
    }
}
