use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::base::{CheckBounds, ReadTileInfo};

#[cube]
pub(crate) trait HorizontalReader<F: Float>: Send + Sync + 'static {
    fn read_horizontal_unchecked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F;

    fn read_horizontal_checked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        check_bounds: CheckBounds,
        read_info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F;
}

#[cube]
pub(crate) trait VerticalReader<F: Float>: Send + Sync + 'static {
    fn read_vertical_unchecked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        gm_stride: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F;

    fn read_vertical_checked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        gm_stride: UInt,
        check_bounds: CheckBounds,
        info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F;
}

#[derive(new)]
/// When vectorization == tile_size
pub(crate) struct MatchingVectorReader;

/// When vectorization != tile_size
#[derive(new)]
pub(crate) struct UnmatchingVectorReader;

#[cube]
impl<F: Float> HorizontalReader<F> for MatchingVectorReader {
    fn read_horizontal_unchecked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        _config: Comptime<CubeTiling2dConfig>,
    ) -> F {
        tensor[gm_position]
    }

    fn read_horizontal_checked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        check_bounds: CheckBounds,
        read_info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F {
        let tile_size = Comptime::map(config, |c| c.tile_size);

        let mut vector = F::vectorized(0., Comptime::get(tile_size));
        if check_bounds.dim_horizontal > read_info.read_col {
            vector = tensor[gm_position];
        }

        vector
    }
}

#[cube]
impl<F: Float> HorizontalReader<F> for UnmatchingVectorReader {
    fn read_horizontal_unchecked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let unroll = Comptime::map(config, |c| c.unroll_tile);
        let vectorization_factor = Comptime::vectorization(tensor);
        let is_scalar = Comptime::map(vectorization_factor, |v| v.val == 1);

        let mut vector = F::vectorized(0., Comptime::get(tile_size));

        for i in range(
            0u32,
            Comptime::get(tile_size / vectorization_factor),
            unroll,
        ) {
            let runtime_vectorization = Comptime::runtime(vectorization_factor);

            if Comptime::get(is_scalar) {
                vector[i] = tensor[gm_position + i];
            } else {
                let intermediate = tensor[gm_position / runtime_vectorization + i];

                for j in range(0u32, Comptime::get(vectorization_factor), unroll) {
                    vector[i * runtime_vectorization + j] = intermediate[j];
                }
            }
        }

        vector
    }

    fn read_horizontal_checked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        check_bounds: CheckBounds,
        read_info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let unroll = Comptime::map(config, |c| c.unroll_tile);
        let vectorization_factor = Comptime::vectorization(tensor);
        let is_scalar = Comptime::map(vectorization_factor, |v| v.val == 1);
        let runtime_vectorization = Comptime::runtime(vectorization_factor);

        let mut vector = F::vectorized(0., Comptime::get(tile_size));

        let mut num_loops = UInt::new(0);
        if check_bounds.dim_horizontal > read_info.read_col {
            let num_reads = UInt::min(
                check_bounds.dim_horizontal - read_info.read_col,
                Comptime::runtime(tile_size),
            );
            num_loops = num_reads / runtime_vectorization;
        }

        for i in range(0u32, num_loops, Comptime::new(false)) {
            if Comptime::get(is_scalar) {
                vector[i] = tensor[gm_position + i];
            } else {
                let intermediate = tensor[gm_position / runtime_vectorization + i];

                for j in range(0u32, Comptime::get(vectorization_factor), unroll) {
                    vector[i * runtime_vectorization + j] = intermediate[j];
                }
            }
        }

        vector
    }
}

#[cube]
impl<F: Float> VerticalReader<F> for UnmatchingVectorReader {
    fn read_vertical_unchecked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        gm_stride: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let unroll = Comptime::map(config, |c| c.unroll_tile);

        let mut vertical = F::vectorized_empty(Comptime::get(tile_size));
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            vertical[i] = tensor[gm_position + i * gm_stride];
        }

        vertical
    }

    fn read_vertical_checked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        gm_stride: UInt,
        check_bounds: CheckBounds,
        info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F {
        let tile_size = Comptime::map(config, |c| c.tile_size);

        let mut vertical = F::vectorized_empty(Comptime::get(tile_size));

        let mut num_reads = UInt::new(0);
        let row = check_bounds.skip_row + info.read_row;
        let dim_vertical = check_bounds.dim_vertical;
        if dim_vertical > row {
            num_reads = UInt::min(dim_vertical - row, Comptime::runtime(tile_size));
        }

        for i in range(0u32, num_reads, Comptime::new(false)) {
            vertical[i] = tensor[gm_position + i * gm_stride];
        }
        for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
            vertical[i] = F::new(0.);
        }

        vertical
    }
}
