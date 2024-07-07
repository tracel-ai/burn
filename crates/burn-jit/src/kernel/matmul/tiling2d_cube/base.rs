use std::cmp::max;

use burn_cube::{prelude::*, Compiler};

use crate::{
    kernel::{
        into_contiguous,
        matmul::{
            config::{tiling2d_cube_count, tiling2d_cube_dim, CubeTiling2dConfig, Tiling2dConfig},
            tiling2d_cube::{
                direct::{base::DirectLoader, loader::WhollyCheckedLoad},
                tile::tile_loading::TileLoader,
            },
        },
    },
    tensor::{JitTensor, MemoryLayout},
    FloatElement, JitRuntime,
};

use super::{
    block_loop::{block_loop, block_loop_expand},
    load_shared_memory::SharedMemoryLoader,
};

#[cube(launch)]
#[allow(unused_mut)]
fn tiling2d_cube<F: Float, S: SharedMemoryLoader<F>>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let dims = get_dims::<F>(lhs, rhs);
    let coordinates = calculate_coordinates(CUBE_POS_X, CUBE_POS_Y, UNIT_POS, config);
    let offsets = calculate_batch_offsets::<F>(lhs, rhs, out, CUBE_POS_Z);
    let shared_memories = make_shared_memories::<F>(config);
    block_loop::<F, S>(
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

    let check_layout = |tensor: JitTensor<R, E, D>| match tensor.memory_layout() {
        MemoryLayout::Contiguous => (tensor, false),
        MemoryLayout::MildlyPermuted {
            transposed,
            batch_swap: _,
        } => (tensor, transposed),
        MemoryLayout::HighlyPermuted => (into_contiguous(tensor), false),
    };
    let (lhs, lhs_transposed) = check_layout(lhs);
    let (rhs, rhs_transposed) = check_layout(rhs);

    let vectorization = |shape: usize| {
        [4, 2]
            .into_iter()
            .filter(|v| shape % v == 0)
            .map(|v| v as u8)
            .next()
            .unwrap_or(1)
    };
    let mut lhs_vectorization = match lhs_transposed {
        true => vectorization(m),
        false => vectorization(k),
    };
    let mut rhs_vectorization = match rhs_transposed {
        true => vectorization(k),
        false => vectorization(n),
    };
    let out_vectorization = vectorization(n);

    let cube_count = tiling2d_cube_count::<R, D>(&out.shape, &config);
    let cube_dim = tiling2d_cube_dim(&config);
    let cube_config = CubeTiling2dConfig::new(&config, m, k, n, lhs_transposed, rhs_transposed);

    let direct = true;
    if direct {
        if lhs_transposed {
            assert!(lhs_vectorization == 4);
        } else {
            lhs_vectorization = 1;
        }
        if rhs_transposed {
            rhs_vectorization = 1;
        } else {
            assert!(rhs_vectorization == 4);
        }
        tiling2d_cube_launch::<
            E::FloatPrimitive,
            DirectLoader<E::FloatPrimitive, WhollyCheckedLoad, WhollyCheckedLoad>,
            R,
        >(
            client,
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                lhs_vectorization,
                &lhs.handle,
                &lhs.strides,
                &lhs.shape.dims,
            ),
            TensorArg::vectorized(
                rhs_vectorization,
                &rhs.handle,
                &rhs.strides,
                &rhs.shape.dims,
            ),
            TensorArg::vectorized(
                out_vectorization,
                &out.handle,
                &out.strides,
                &out.shape.dims,
            ),
            cube_config,
        );
    } else {
        let lhs_vectorization = match lhs_transposed {
            true => vectorization(m),
            false => vectorization(k),
        };
        tiling2d_cube_launch::<E::FloatPrimitive, TileLoader, R>(
            client,
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                lhs_vectorization,
                &lhs.handle,
                &lhs.strides,
                &lhs.shape.dims,
            ),
            TensorArg::vectorized(
                rhs_vectorization,
                &rhs.handle,
                &rhs.strides,
                &rhs.shape.dims,
            ),
            TensorArg::vectorized(
                out_vectorization,
                &out.handle,
                &out.strides,
                &out.shape.dims,
            ),
            cube_config,
        );
    }
    out
}
