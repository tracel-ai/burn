use burn_cube::prelude::*;

use crate::{
    kernel::{into_contiguous, matmul::Tiling2dConfig},
    tensor::JitTensor,
    FloatElement, JitRuntime,
};

use super::{
    config::CubeTiling2dConfig,
    tiling2d_core::{tiling2d_core, tiling2d_core_expand},
};
use crate::kernel::matmul::tiling2d_launch_options;

#[cube(launch)]
#[allow(unused_mut)]
fn tiling2d_cube<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    mut out: Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let coordinates = calculate_coordinates(config);
    let offsets = calculate_offsets::<F>(lhs, rhs, out, coordinates, config);
    let shared_memories = make_shared_memories::<F>(config);
    tiling2d_core(lhs, rhs, out, coordinates, offsets, shared_memories, config);
}

#[derive(CubeType)]
pub(crate) struct SharedMemories<F: Float> {
    pub lhs: SharedMemory<F>,
    pub rhs: SharedMemory<F>,
}

#[derive(CubeType)]
pub(crate) struct Offsets {
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
fn calculate_coordinates(config: Comptime<CubeTiling2dConfig>) -> Coordinates {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let tile_size = Comptime::map(config, |c| c.tile_size);

    // Topology info
    let n_threads_per_row = ((Comptime::runtime(block_size_n) - UInt::new(1))
        / Comptime::runtime(tile_size))
        + UInt::new(1);

    // Cube offset
    let skip_row = CUBE_POS_X * Comptime::runtime(block_size_m);
    let skip_col = CUBE_POS_Y * Comptime::runtime(block_size_n);

    // Position of the first element of the unit, relative to the cube
    let unit_row = (UNIT_POS / n_threads_per_row) * Comptime::runtime(tile_size);
    let unit_col = (UNIT_POS % n_threads_per_row) * Comptime::runtime(tile_size);

    Coordinates {
        unit_row,
        unit_col,
        skip_row,
        skip_col,
    }
}

#[cube]
#[allow(unused_mut)]
fn calculate_offsets<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    mut out: Tensor<F>,
    coordinates: Coordinates,
    config: Comptime<CubeTiling2dConfig>,
) -> Offsets {
    let unroll = Comptime::map(config, |c| c.unroll);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let rank = out.rank();

    let dim_m = lhs.shape(rank - UInt::new(2));
    let dim_n = rhs.shape(rank - UInt::new(1));

    // Batch offset for output
    let batch = ABSOLUTE_POS_Z;
    let offset_output = dim_m * dim_n * batch / Comptime::runtime(tile_size);

    // Calculate offset for lhs and rhs, without regards to batches
    let mut offset_lhs = coordinates.skip_row * lhs.stride(rank - UInt::new(2));
    let mut offset_rhs = coordinates.skip_col;

    // Batch offset for lhs, rhs
    for b in range(0u32, rank - UInt::new(2), unroll) {
        let tmp = offset_output / out.stride(b);
        offset_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        offset_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    offset_lhs /= Comptime::runtime(tile_size);
    offset_rhs /= Comptime::runtime(tile_size);

    Offsets {
        lhs: offset_lhs,
        rhs: offset_rhs,
        out: offset_output,
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

/// Matrix multiplication using tiling 2d algorithm with
/// written in Cube
pub fn matmul_tiling_2d_cube<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    mut config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    let m = lhs.shape.dims[D - 2];
    let k = lhs.shape.dims[D - 1];
    let n = rhs.shape.dims[D - 1];

    let client = lhs.client.clone();

    let lhs = match lhs.batch_swapped_with_row_col() {
        true => into_contiguous(lhs),
        false => lhs,
    };
    let rhs = match rhs.batch_swapped_with_row_col() {
        true => into_contiguous(rhs),
        false => rhs,
    };

    config.block_size_m = 16;
    config.block_size_n = 16;
    config.block_size_k = 16; // k must be <= both m and n
    let cube_count = tiling2d_launch_options(&out.shape, config.clone());

    let vectorization_factor = 1;
    let x = (config.block_size_m / vectorization_factor) as u32;
    let y = (config.block_size_n / vectorization_factor) as u32;
    let settings = KernelSettings::default()
        .vectorize_input(0, vectorization_factor as u8)
        .vectorize_input(1, vectorization_factor as u8)
        .vectorize_output(0, vectorization_factor as u8)
        .cube_dim(CubeDim { x, y, z: 1 });

    tiling2d_cube_launch::<E::CubeElement, R>(
        client,
        cube_count,
        settings,
        TensorHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
        TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
        CubeTiling2dConfig::new(config, m, k, n, vectorization_factor as usize),
    );

    out
}
