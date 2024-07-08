use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::loader::{CheckBounds, ReadTileInfo};

#[cube]
pub(crate) trait ContiguousAccess<F: Float>: Send + Sync + 'static {
    fn read_contiguous_unchecked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F;

    fn read_contiguous_checked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        check_bounds: CheckBounds,
        read_info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F;

    fn write_contiguous_unchecked(
        out: &mut Tensor<F>,
        out_position: UInt,
        results: &Array<F>,
        results_position: UInt,
        config: Comptime<CubeTiling2dConfig>,
    );

    fn write_contiguous_checked(
        out: &mut Tensor<F>,
        out_position: UInt,
        results: &Array<F>,
        results_position: UInt,
        check_bounds: CheckBounds,
        write_col: UInt,
        config: Comptime<CubeTiling2dConfig>,
    );
}

#[cube]
pub(crate) trait StridedAccess<F: Float>: Send + Sync + 'static {
    fn read_strided_unchecked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        gm_stride: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F;

    fn read_strided_checked(
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
pub(crate) struct MatchingVectorization;

/// When vectorization != tile_size
#[derive(new)]
pub(crate) struct UnmatchingVectorization;

#[cube]
impl<F: Float> ContiguousAccess<F> for MatchingVectorization {
    fn read_contiguous_unchecked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        _config: Comptime<CubeTiling2dConfig>,
    ) -> F {
        tensor[gm_position]
    }

    fn read_contiguous_checked(
        tensor: &Tensor<F>,
        gm_position: UInt,
        _check_bounds: CheckBounds,
        _read_info: ReadTileInfo,
        config: Comptime<CubeTiling2dConfig>,
    ) -> F {
        // If vectorization matches, then it's certain to fit since tile_size divides block_sizes
        MatchingVectorization::read_contiguous_unchecked(tensor, gm_position, config)
    }

    fn write_contiguous_unchecked(
        out: &mut Tensor<F>,
        out_position: UInt,
        results: &Array<F>,
        results_position: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let unroll = Comptime::map(config, |c| c.unroll_tile);

        let mut output_elem = F::vectorized_empty(Comptime::get(tile_size));

        for i in range(0u32, Comptime::get(tile_size), unroll) {
            output_elem[i] = results[results_position + i];
        }

        out[out_position / Comptime::runtime(tile_size)] = output_elem;
    }

    fn write_contiguous_checked(
        out: &mut Tensor<F>,
        out_position: UInt,
        results: &Array<F>,
        results_position: UInt,
        _check_bounds: CheckBounds,
        _write_col: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        // If vectorization matches, then it's certain to fit since tile_size divides block_sizes
        MatchingVectorization::write_contiguous_unchecked(
            out,
            out_position,
            results,
            results_position,
            config,
        )
    }
}

#[cube]
impl<F: Float> ContiguousAccess<F> for UnmatchingVectorization {
    fn read_contiguous_unchecked(
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
                let intermediate = tensor[gm_position + i];

                for j in range(0u32, Comptime::get(vectorization_factor), unroll) {
                    vector[i * runtime_vectorization + j] = intermediate[j];
                }
            }
        }

        vector
    }

    fn read_contiguous_checked(
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
                let intermediate = tensor[gm_position + i];

                for j in range(0u32, Comptime::get(vectorization_factor), unroll) {
                    vector[i * runtime_vectorization + j] = intermediate[j];
                }
            }
        }

        vector
    }

    fn write_contiguous_unchecked(
        out: &mut Tensor<F>,
        out_position: UInt,
        results: &Array<F>,
        results_position: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let unroll = Comptime::map(config, |c| c.unroll_tile);
        let vectorization_factor = Comptime::vectorization(out);
        let runtime_vectorization = Comptime::runtime(vectorization_factor);
        let is_scalar = Comptime::map(vectorization_factor, |v| v.val == 1);

        for i in range(
            0u32,
            Comptime::get(tile_size / vectorization_factor),
            unroll,
        ) {
            if Comptime::get(is_scalar) {
                out[i + out_position] = results[results_position + i];
            } else {
                let mut output_elem = F::vectorized_empty(Comptime::get(vectorization_factor));

                for j in range(0u32, Comptime::get(vectorization_factor), unroll) {
                    let index = i * runtime_vectorization + j;
                    output_elem[j] = results[results_position + index];
                }

                out[i + out_position / runtime_vectorization] = output_elem;
            }
        }
    }

    fn write_contiguous_checked(
        out: &mut Tensor<F>,
        out_position: UInt,
        results: &Array<F>,
        results_position: UInt,
        check_bounds: CheckBounds,
        write_col: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let vectorization_factor = Comptime::vectorization(out);
        let runtime_vectorization = Comptime::runtime(vectorization_factor);
        let is_scalar = Comptime::map(vectorization_factor, |v| v.val == 1);

        let mut num_loops = UInt::new(0);
        if check_bounds.dim_horizontal > write_col {
            let num_writes = UInt::min(
                check_bounds.dim_horizontal - write_col,
                Comptime::runtime(tile_size),
            );
            num_loops = num_writes / runtime_vectorization;
        }

        for i in range(0u32, num_loops, Comptime::new(false)) {
            let unroll = Comptime::map(config, |c| c.unroll_tile);

            if Comptime::get(is_scalar) {
                out[i + out_position] = results[results_position + i];
            } else {
                let mut output_elem = F::vectorized_empty(Comptime::get(vectorization_factor));

                for j in range(0u32, Comptime::get(vectorization_factor), unroll) {
                    let index = i * runtime_vectorization + j;
                    output_elem[j] = results[results_position + index];
                }

                out[i + out_position / runtime_vectorization] = output_elem;
            }
        }
    }
}

#[cube]
impl<F: Float> StridedAccess<F> for UnmatchingVectorization {
    fn read_strided_unchecked(
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

    fn read_strided_checked(
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
