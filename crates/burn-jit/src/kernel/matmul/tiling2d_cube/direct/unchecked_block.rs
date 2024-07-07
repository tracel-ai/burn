use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::{
    base::{CheckBounds, Loader, ReadTileInfo},
    vector_reader::{HorizontalReader, UnmatchingVectorReader, VerticalReader},
};

#[derive(new)]
/// Assumes block sizes divide tensor shape
pub(crate) struct UncheckedBlockLoad<H> {
    horizontal_reader: H,
}

#[cube]
impl<F: Float, V: HorizontalReader<F>> Loader<F> for UncheckedBlockLoad<V> {
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
            let gm_position =
                (info.gm_position_base + i * info.gm_stride) / Comptime::runtime(vectorization);
            let sm_position =
                (info.sm_position_base + i * info.sm_stride) / Comptime::runtime(tile_size);

            shared_memory[sm_position] = V::read_horizontal_unchecked(tensor, gm_position, config);
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

            shared_memory[sm_position] = UnmatchingVectorReader::read_vertical_unchecked(
                tensor,
                gm_position,
                info.gm_stride,
                config,
            );
        }
    }
}
