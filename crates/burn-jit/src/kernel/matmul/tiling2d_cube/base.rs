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
const TILE_SIZE: usize = 4;

#[cube(launch)]
#[allow(unused_mut)]
fn tiling2d_cube<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    mut out: Tensor<F>,
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
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    mut out: Tensor<F>,
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
    mut config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    let m = lhs.shape.dims[D - 2];
    let k = lhs.shape.dims[D - 1];
    let n = rhs.shape.dims[D - 1];

    let client = lhs.client.clone();

    let lhs = into_contiguous(lhs);
    let rhs = into_contiguous(rhs);

    let vectorization = |shape: usize| {
        [4, 2, 1]
            .into_iter()
            .filter(|v| shape % v == 0)
            .map(|v| v as u8)
            .next()
            .unwrap()
    };

    config.block_size_m = 64;
    config.block_size_n = 64;
    config.block_size_k = 32; // k must be <= both m and n
    let cube_count = tiling2d_launch_options(&out.shape, config.clone());

    let x = (config.block_size_m / TILE_SIZE) as u32;
    let y = (config.block_size_n / TILE_SIZE) as u32;

    let settings = KernelSettings::default()
        .vectorize_input(0, vectorization(m))
        .vectorize_input(1, vectorization(k))
        .vectorize_output(0, vectorization(n))
        .cube_dim(CubeDim { x, y, z: 1 });

    tiling2d_cube_launch::<E::CubeElement, R>(
        client,
        cube_count,
        settings,
        TensorHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
        TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
        CubeTiling2dConfig::new(config, m, k, n, TILE_SIZE as usize),
    );

    out
}

#[cfg(feature = "export_tests")]
/// Exported tests for tiling2d
pub mod tests {
    use crate::JitBackend;

    use super::*;

    #[cube(launch)]
    #[allow(unused_mut)]
    fn calculate_offsets_test<F: Float>(
        lhs: Tensor<F>,
        rhs: Tensor<F>,
        out: Tensor<F>,
        num_batch: UInt,
        mut offsets: Array<F>,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        // So all bindings are read
        let _x = lhs[0];
        let _x = rhs[0];
        let _x = out[0];

        let calculated = calculate_batch_offsets::<F>(lhs, rhs, out, num_batch, config);

        offsets[0] = F::cast_from(calculated.lhs);
        offsets[1] = F::cast_from(calculated.rhs);
        offsets[2] = F::cast_from(calculated.out);
    }

    /// Exported test
    pub fn calculate_offsets_unit_test<R: JitRuntime>(device: &R::Device) {
        pub type B<R> = JitBackend<R, f32, i32>;

        let tile_size = 4;
        let b = 4;
        let (m, k, n) = (3, 4, 5);
        let lhs = burn_tensor::Tensor::<B<R>, 3>::zeros([b, m, k], device).into_primitive();
        let rhs = burn_tensor::Tensor::<B<R>, 3>::zeros([b, k, n], device).into_primitive();
        let out = burn_tensor::Tensor::<B<R>, 3>::zeros([b, m, n], device).into_primitive();

        let client = R::client(device);
        let offsets = client.empty(3 * core::mem::size_of::<f32>());

        // Unit test
        let cube_count = CubeCount::new(1, 1, 1);
        let settings = KernelSettings::default()
            .cube_dim(CubeDim::new(1, 1, 1))
            .vectorize_input(0, tile_size as u8)
            .vectorize_input(1, tile_size as u8)
            .vectorize_input(2, tile_size as u8);

        let mut tiling2d_config = Tiling2dConfig::default();
        tiling2d_config.block_size_m = 8;
        tiling2d_config.block_size_k = 8;
        tiling2d_config.block_size_n = 8;
        let config = CubeTiling2dConfig::new(tiling2d_config, 8, 8, 8, tile_size);

        calculate_offsets_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
            b as u32,
            ArrayHandle::new(&offsets, 3),
            config,
        );

        let actual = client.read(offsets.binding()).read_sync().unwrap();
        let actual = f32::from_bytes(&actual);
        let expected = &[0.0, 0.0, 60.0];
        assert_eq!(actual, expected);
    }
}
