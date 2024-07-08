use std::marker::PhantomData;

use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::{
    base::{
        all_zeros_comptime, all_zeros_comptime_expand, all_zeros_runtime, all_zeros_runtime_expand,
        CheckBounds, Loader, ReadTileInfo,
    },
    vector_reader::{HorizontalReader, UnmatchingVectorReader, VerticalReader},
};

pub(crate) struct HorizontalBlockCheckLoad<H> {
    _h: PhantomData<H>,
}

#[cube]
impl<F: Float, V: HorizontalReader<F>> Loader<F> for HorizontalBlockCheckLoad<V> {
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

                shared_memory[sm_position] =
                    V::read_horizontal_checked(tensor, gm_position, check_bounds, info, config);
            }
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

            shared_memory[sm_position] = UnmatchingVectorReader::read_vertical_unchecked(
                tensor,
                gm_position,
                info.gm_stride,
                config,
            );
        }

        all_zeros_runtime(
            shared_memory,
            num_reads,
            info.sm_position_base,
            info.sm_stride,
            config,
        );
    }
}
