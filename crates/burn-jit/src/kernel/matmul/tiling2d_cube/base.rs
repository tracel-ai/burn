use cubecl::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::block_loop::block_loop;

#[cube(launch)]
#[allow(unused_mut)]
pub fn tiling2d_cube_kernel<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let dims = get_dims::<F>(lhs, rhs);
    let coordinates = calculate_coordinates(CUBE_POS_X, CUBE_POS_Y, UNIT_POS, config);
    let offsets = calculate_batch_offsets::<F>(lhs, rhs, out, CUBE_POS_Z);
    let shared_memories = make_shared_memories::<F>(config);
    block_loop::<F>(
        lhs,
        rhs,
        out,
        coordinates,
        offsets,
        shared_memories,
        config,
        dims,
    );
}

#[derive(CubeType, Copy, Clone)]
/// Information available at runtime only
/// Strides assume contiguous
pub(crate) struct Dimensions {
    pub m: UInt,
    pub k: UInt,
    pub n: UInt,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct SharedMemories<F: Float> {
    pub lhs: SharedMemory<F>,
    pub rhs: SharedMemory<F>,
}

#[derive(CubeType, Copy, Clone)]
/// Number of elements in previous batches
/// Not divided by vectorization facto
pub(crate) struct BatchOffsets {
    pub lhs: UInt,
    pub rhs: UInt,
    pub out: UInt,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct Coordinates {
    pub unit_row: UInt,
    pub unit_col: UInt,
    pub skip_row: UInt,
    pub skip_col: UInt,
}

#[cube]
fn get_dims<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>) -> Dimensions {
    let rank = lhs.rank();
    let first_dim = rank - UInt::new(2);
    let second_dim = rank - UInt::new(1);
    let m = lhs.shape(first_dim);
    let k = lhs.shape(second_dim);
    let n = rhs.shape(second_dim);

    Dimensions { m, k, n }
}

#[cube]
fn calculate_coordinates(
    cube_pos_x: UInt,
    cube_pos_y: UInt,
    unit_pos: UInt,
    config: Comptime<CubeTiling2dConfig>,
) -> Coordinates {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let tile_size = Comptime::map(config, |c| c.tile_size);

    let n_units_per_row = ((Comptime::runtime(block_size_n) - UInt::new(1))
        / Comptime::runtime(tile_size))
        + UInt::new(1);

    // Cube offset
    let skip_row = cube_pos_x * Comptime::runtime(block_size_m);
    let skip_col = cube_pos_y * Comptime::runtime(block_size_n);

    // Position of the first element of the unit, relative to the cube
    let unit_row = (unit_pos / n_units_per_row) * Comptime::runtime(tile_size);
    let unit_col = (unit_pos % n_units_per_row) * Comptime::runtime(tile_size);

    Coordinates {
        unit_row,
        unit_col,
        skip_row,
        skip_col,
    }
}

#[cube]
#[allow(unused_mut)]
fn calculate_batch_offsets<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &Tensor<F>,
    batch_number: UInt,
) -> BatchOffsets {
    let rank = out.rank();

    let dim_m = lhs.shape(rank - UInt::new(2));
    let dim_n = rhs.shape(rank - UInt::new(1));

    // Batch offset for output
    let mut offset_out = dim_m * dim_n * batch_number;
    let mut offset_lhs = UInt::new(0);
    let mut offset_rhs = UInt::new(0);

    // Batch offset for lhs, rhs
    for b in range(0u32, rank - UInt::new(2), Comptime::new(false)) {
        let tmp = offset_out / out.stride(b);
        offset_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        offset_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    BatchOffsets {
        lhs: offset_lhs,
        rhs: offset_rhs,
        out: offset_out,
    }
}

#[cube]
fn make_shared_memories<F: Float>(config: Comptime<CubeTiling2dConfig>) -> SharedMemories<F> {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);

    let lhs = SharedMemory::<F>::vectorized(
        Comptime::get(block_size_k * block_size_m / tile_size),
        Comptime::get(tile_size),
    );

    let rhs = SharedMemory::<F>::vectorized(
        Comptime::get(block_size_k * block_size_n / tile_size),
        Comptime::get(tile_size),
    );

    SharedMemories { lhs, rhs }
}
