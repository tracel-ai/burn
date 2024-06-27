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

// Other tile sizes are not supported
pub(crate) const TILE_SIZE: usize = 4;

#[cube(launch)]
#[allow(unused_mut)]
fn tiling2d_cube<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let coordinates = calculate_coordinates(CUBE_POS_X, CUBE_POS_Y, UNIT_POS, config);
    let offsets = calculate_batch_offsets::<F>(lhs, rhs, out, CUBE_POS_Z, config);
    let shared_memories = make_shared_memories::<F>(config);
    tiling2d_core(lhs, rhs, out, coordinates, offsets, shared_memories, config);
}

#[derive(CubeType)]
pub(crate) struct SharedMemories<F: Float> {
    pub lhs: SharedMemory<F>,
    pub rhs: SharedMemory<F>,
}

#[derive(CubeType)]
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
    config: Comptime<CubeTiling2dConfig>,
) -> BatchOffsets {
    let unroll = Comptime::map(config, |c| c.unroll);
    let rank = out.rank();

    let dim_m = lhs.shape(rank - UInt::new(2));
    let dim_n = rhs.shape(rank - UInt::new(1));

    // Batch offset for output
    let mut offset_out = dim_m * dim_n * batch_number;
    let mut offset_lhs = UInt::new(0);
    let mut offset_rhs = UInt::new(0);

    // Batch offset for lhs, rhs
    for b in range(0u32, rank - UInt::new(2), unroll) {
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

/// Matrix multiplication using tiling 2d algorithm with
/// written in Cube
pub fn matmul_tiling_2d_cube<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    let m = lhs.shape.dims[D - 2];
    let k = lhs.shape.dims[D - 1];
    let n = rhs.shape.dims[D - 1];

    let client = lhs.client.clone();

    let lhs = into_contiguous(lhs);
    let rhs = into_contiguous(rhs);

    let vectorization = |shape: usize| {
        [4, 2]
            .into_iter()
            .filter(|v| shape % v == 0)
            .map(|v| v as u8)
            .next()
            .unwrap_or(1)
    };

    let cube_config = CubeTiling2dConfig::new(&config, m, k, n, TILE_SIZE as usize);
    let cube_count = tiling2d_launch_options(&out.shape, &config);

    let x = config.grid_x as u32;
    let y = config.grid_y as u32;

    let settings = KernelSettings::default()
        .vectorize_input(0, vectorization(k))
        .vectorize_input(1, vectorization(n))
        .vectorize_output(0, vectorization(n))
        .cube_dim(CubeDim { x, y, z: 1 });

    tiling2d_cube_launch::<E::FloatPrimitive, R>(
        client,
        cube_count,
        settings,
        TensorHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
        TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
        cube_config,
    );

    out
}
