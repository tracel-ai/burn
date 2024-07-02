use burn_cube::prelude::*;

use super::{
    base::{BatchOffsets, Coordinates, CubeTiling2dInfo, SharedMemories},
    compute_loop::{compute_loop, compute_loop_expand},
    config::CubeTiling2dConfig,
    load_shared_memory::{
        load_lhs_transposed, load_lhs_transposed_expand, load_rhs_plain, load_rhs_plain_expand,
    },
    write_output::{write_to_output, write_to_output_expand},
};

#[cube]
#[allow(unused_mut)]
pub(crate) fn tiling2d_core<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    coordinates: Coordinates,
    offsets: BatchOffsets,
    shared: SharedMemories<F>,
    config: Comptime<CubeTiling2dConfig>,
    info: CubeTiling2dInfo,
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let mut results = init_results::<F>(config);

    let n_loops = calculate_n_loops::<F>(lhs.shape(lhs.rank() - UInt::new(1)), config);

    for k in range(0u32, n_loops, Comptime::new(false)) {
        let k = k * Comptime::runtime(block_size_k);

        load_lhs_transposed::<F>(lhs, coordinates, k, offsets.lhs, shared.lhs, config, info);
        load_rhs_plain::<F>(rhs, coordinates, k, offsets.rhs, shared.rhs, config, info);

        sync_units();

        compute_loop::<F>(coordinates, shared.lhs, shared.rhs, &mut results, config);

        sync_units();
    }

    write_to_output::<F>(
        out,
        &results,
        coordinates,
        offsets.out,
        info.out_stride,
        config,
    );
}

#[cube]
fn init_results<F: Float>(config: Comptime<CubeTiling2dConfig>) -> Array<F> {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll);

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
