use std::cmp::max;

use burn_cube::{prelude::*, Compiler};

use crate::{
    kernel::{
        into_contiguous,
        matmul::config::{
            tiling2d_cube_count, tiling2d_cube_dim, CubeTiling2dConfig, Tiling2dConfig,
        },
    },
    tensor::{JitTensor, MemoryLayout},
    FloatElement, JitRuntime,
};

use super::block_loop::{block_loop, block_loop_expand};

#[cube(launch)]
#[allow(unused_mut)]
fn tiling2d_cube<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let info = get_info::<F>(lhs, rhs, out);
    let coordinates = calculate_coordinates(CUBE_POS_X, CUBE_POS_Y, UNIT_POS, config);
    let offsets = calculate_batch_offsets::<F>(lhs, rhs, out, CUBE_POS_Z);
    let shared_memories = make_shared_memories::<F>(config);
    block_loop(
        lhs,
        rhs,
        out,
        coordinates,
        offsets,
        shared_memories,
        config,
        info,
    );
}

#[derive(CubeType, Copy, Clone)]
/// Information available at runtime only
/// Strides assume contiguous
pub(crate) struct CubeTiling2dInfo {
    pub dim_m: UInt,
    pub dim_k: UInt,
    pub dim_n: UInt,
    pub lhs_stride: UInt,
    pub rhs_stride: UInt,
    pub out_stride: UInt,
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
fn get_info<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>, out: &Tensor<F>) -> CubeTiling2dInfo {
    let rank = lhs.rank();
    let first_dim = rank - UInt::new(2);
    let second_dim = rank - UInt::new(1);
    let dim_m = lhs.shape(first_dim);
    let dim_k = lhs.shape(second_dim);
    let dim_n = rhs.shape(second_dim);
    let lhs_stride = lhs.stride(first_dim);
    let rhs_stride = rhs.stride(first_dim);
    let out_stride = out.stride(first_dim);

    CubeTiling2dInfo {
        dim_m,
        dim_k,
        dim_n,
        lhs_stride,
        rhs_stride,
        out_stride,
    }
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

/// Matrix multiplication using tiling 2d algorithm
pub fn matmul_tiling_2d_cube<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    assert!(
        config.block_size_k * max(config.block_size_m, config.block_size_n)
            <= <R::Compiler as Compiler>::max_shared_memory_size(),
        "Shared memory limit will be busted. "
    );

    let m = lhs.shape.dims[D - 2];
    let k = lhs.shape.dims[D - 1];
    let n = rhs.shape.dims[D - 1];

    let client = lhs.client.clone();

    let lhs = match lhs.memory_layout() == MemoryLayout::HighlyPermuted {
        true => into_contiguous(lhs),
        false => lhs,
    };
    let rhs = match lhs.memory_layout() == MemoryLayout::HighlyPermuted {
        true => into_contiguous(rhs),
        false => rhs,
    };

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

    tiling2d_cube_launch::<E::FloatPrimitive, R>(
        client,
        tiling2d_cube_count(&out.shape, &config),
        tiling2d_cube_dim(&config),
        TensorArg::vectorized(vectorization(k), &lhs.handle, &lhs.strides, &lhs.shape.dims),
        TensorArg::vectorized(vectorization(n), &rhs.handle, &rhs.strides, &rhs.shape.dims),
        TensorArg::vectorized(vectorization(n), &out.handle, &out.strides, &out.shape.dims),
        CubeTiling2dConfig::new(&config, m, k, n, false, false),
    );

    out
}
