use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::{
    base::{BatchOffsets, Coordinates, Dimensions, SharedMemories},
    compute_loop::{compute_loop, compute_loop_expand},
    tile::{loader::TileLoader, writer::TileWriter},
    load_shared_memory::{load_to_shared_memories, load_to_shared_memories_expand},
    write_output::{write_to_output, write_to_output_expand},
};

#[cube]
pub(crate) fn block_loop<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    coordinates: Coordinates,
    offsets: BatchOffsets,
    shared: SharedMemories<F>,
    config: Comptime<CubeTiling2dConfig>,
    dims: Dimensions,
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let mut results = init_results::<F>(config);

    let n_loops = calculate_n_loops::<F>(lhs.shape(lhs.rank() - UInt::new(1)), config);

    for k in range(0u32, n_loops, Comptime::new(false)) {
        let k = k * Comptime::runtime(block_size_k);

        load_to_shared_memories::<F, TileLoader<F>>(
            lhs,
            rhs,
            coordinates,
            k,
            offsets,
            shared,
            config,
            dims,
        );

        sync_units();

        compute_loop::<F>(coordinates, shared.lhs, shared.rhs, &mut results, config);

        sync_units();
    }

    write_to_output::<F, TileWriter<F>>(out, &results, coordinates, offsets.out, dims, config);
}

#[cube]
fn init_results<F: Float>(config: Comptime<CubeTiling2dConfig>) -> Array<F> {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll_tile);

    let mut results = Array::<F>::new(Comptime::get(tile_size * tile_size));
    for i in range(0u32, Comptime::get(tile_size * tile_size), unroll) {
        results[i] = F::new(0.);
    }

    results
}

#[cube]
#[allow(unused_assignments)]
fn calculate_n_loops<F: Float>(dim_k: UInt, config: Comptime<CubeTiling2dConfig>) -> UInt {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);

    let mut n_loops = UInt::new(0); // TODO support syntax let x = if ... else ...
    if Comptime::get(check_k_bounds) {
        n_loops = UInt::cast_from(F::ceil(
            F::cast_from(dim_k) / F::cast_from(Comptime::runtime(block_size_k)),
        ));
    } else {
        n_loops = dim_k / Comptime::runtime(block_size_k);
    }

    n_loops
}
