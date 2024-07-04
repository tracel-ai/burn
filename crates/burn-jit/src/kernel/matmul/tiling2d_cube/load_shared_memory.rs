use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::{
    base::{BatchOffsets, Coordinates, CubeTiling2dInfo, SharedMemories},
    tile_read::{read_tile_from_global_memory, read_tile_from_global_memory_expand},
    tile_write::{
        write_tile_plain, write_tile_plain_expand, write_tile_transposed,
        write_tile_transposed_expand,
    },
};

#[cube]
pub(crate) fn load_to_shared_memories<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    coordinates: Coordinates,
    k: UInt,
    offsets: BatchOffsets,
    shared: SharedMemories<F>,
    config: Comptime<CubeTiling2dConfig>,
    info: CubeTiling2dInfo,
) {
    let lhs_transposed = Comptime::map(config, |c| c.lhs_transposed);
    let rhs_transposed = Comptime::map(config, |c| c.rhs_transposed);

    // Lhs must be loaded as transposed. If it already is in global memory, we load as plain.
    if Comptime::get(lhs_transposed) {
        // load_lhs_plain::<F>(lhs, coordinates, k, offsets.lhs, shared.lhs, config, info);
    } else {
        load_lhs_transposed::<F>(lhs, coordinates, k, offsets.lhs, shared.lhs, config, info);
    }

    // Rhs must be loaded as plain. If it is transposed in global memory, we transpose it back.
    if Comptime::get(rhs_transposed) {
        // load_rhs_transposed::<F>(rhs, coordinates, k, offsets.rhs, shared.rhs, config, info);
    } else {
        load_rhs_plain::<F>(rhs, coordinates, k, offsets.rhs, shared.rhs, config, info);
    }
}

#[cube]
pub(crate) fn load_lhs_transposed<F: Float>(
    lhs: &Tensor<F>,
    coordinates: Coordinates,
    k: UInt,
    batch_offset: UInt,
    shared_lhs: SharedMemory<F>,
    config: Comptime<CubeTiling2dConfig>,
    info: CubeTiling2dInfo,
) {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let tile_size = Comptime::map(config, |c| c.tile_size);

    let sm_stride = Comptime::runtime(block_size_m);

    let cube_offset = coordinates.skip_row * info.lhs_stride;
    let offset = cube_offset + k + batch_offset;

    let mut tile = Array::<F>::vectorized(Comptime::get(tile_size), Comptime::get(tile_size));

    read_tile_from_global_memory::<F>(
        lhs,
        &mut tile,
        offset,
        coordinates.unit_row,
        coordinates.unit_col,
        coordinates.skip_row,
        k,
        info.lhs_stride,
        info.dim_m,
        info.dim_k,
        Comptime::map(config, |c| c.check_m_bounds),
        Comptime::map(config, |c| c.check_k_bounds),
        config,
    );

    write_tile_transposed::<F>(
        &tile,
        shared_lhs,
        coordinates.unit_col,
        coordinates.unit_row,
        sm_stride,
        config,
    );
}

#[cube]
pub(crate) fn load_rhs_plain<F: Float>(
    rhs: &Tensor<F>,
    coordinates: Coordinates,
    k: UInt,
    batch_offset: UInt,
    shared_rhs: SharedMemory<F>,
    config: Comptime<CubeTiling2dConfig>,
    info: CubeTiling2dInfo,
) {
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let tile_size = Comptime::map(config, |c| c.tile_size);

    let sm_stride = Comptime::runtime(block_size_n);

    let offset = coordinates.skip_col + k * info.rhs_stride + batch_offset;

    let mut tile = Array::<F>::vectorized(Comptime::get(tile_size), Comptime::get(tile_size));

    read_tile_from_global_memory::<F>(
        rhs,
        &mut tile,
        offset,
        coordinates.unit_row,
        coordinates.unit_col,
        k,
        coordinates.skip_col,
        info.rhs_stride,
        info.dim_k,
        info.dim_n,
        Comptime::map(config, |c| c.check_k_bounds),
        Comptime::map(config, |c| c.check_n_bounds),
        config,
    );

    write_tile_plain::<F>(
        &tile,
        shared_rhs,
        coordinates.unit_row,
        coordinates.unit_col,
        sm_stride,
        config,
    );
}

#[cfg(feature = "export_tests")]
/// Exported tests for loading to shared memory
pub mod tests {
    use crate::kernel::matmul::tiling2d_cube::test_utils::{
        assert_equals, create_empty, make_config, range_tensor, TILE_SIZE,
    };
    use crate::JitRuntime;

    use super::{super::base::CoordinatesExpand, super::base::CubeTiling2dInfoExpand, *};

    #[cube(launch)]
    fn load_tensor_test<F: Float>(
        tensor: &Tensor<F>,
        sm_out: &mut Array<F>,
        unit_row: UInt,
        unit_col: UInt,
        k: UInt,
        config: Comptime<CubeTiling2dConfig>,
        is_lhs: Comptime<bool>,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let block_size_k = Comptime::map(config, |c| c.block_size_k);
        let block_size_m = Comptime::map(config, |c| c.block_size_m);
        let sm_size = block_size_k * block_size_m / tile_size;
        let shared_memory =
            SharedMemory::<F>::vectorized(Comptime::get(sm_size), Comptime::get(tile_size));

        let offset = UInt::new(0);

        let coordinates = Coordinates {
            unit_row,
            unit_col,
            skip_row: UInt::new(0),
            skip_col: UInt::new(0),
        };

        if Comptime::get(is_lhs) {
            let info = CubeTiling2dInfo {
                dim_m: tensor.shape(tensor.rank() - UInt::new(2)),
                dim_k: tensor.shape(tensor.rank() - UInt::new(1)),
                dim_n: UInt::new(0),
                lhs_stride: tensor.stride(tensor.rank() - UInt::new(2)),
                rhs_stride: UInt::new(0),
                out_stride: UInt::new(0),
            };

            load_lhs_transposed(tensor, coordinates, k, offset, shared_memory, config, info);
        } else {
            let info = CubeTiling2dInfo {
                dim_m: UInt::new(0),
                dim_k: tensor.shape(tensor.rank() - UInt::new(2)),
                dim_n: tensor.shape(tensor.rank() - UInt::new(1)),
                lhs_stride: UInt::new(0),
                rhs_stride: tensor.stride(tensor.rank() - UInt::new(2)),
                out_stride: UInt::new(0),
            };

            load_rhs_plain(tensor, coordinates, k, offset, shared_memory, config, info);
        }

        for i in range(0u32, Comptime::get(sm_size), Comptime::new(false)) {
            sm_out[i] = shared_memory[i];
        }
    }

    #[cube(launch)]
    fn load_tensor_multiple_tiles_test<F: Float>(
        tensor: &Tensor<F>,
        sm_out: &mut Array<F>,
        k: UInt,
        config: Comptime<CubeTiling2dConfig>,
        is_lhs: Comptime<bool>,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let block_size_k = Comptime::map(config, |c| c.block_size_k);
        let block_size_m = Comptime::map(config, |c| c.block_size_m);
        let sm_size = block_size_k * block_size_m / tile_size;
        let shared_memory =
            SharedMemory::<F>::vectorized(Comptime::get(sm_size), Comptime::get(tile_size));

        let unit_row = UInt::new(4) * UNIT_POS_X;
        let unit_col = UInt::new(4) * UNIT_POS_Y;
        let offset = UInt::new(0);

        let coordinates = Coordinates {
            unit_row,
            unit_col,
            skip_row: UInt::new(0),
            skip_col: UInt::new(0),
        };

        if Comptime::get(is_lhs) {
            let info = CubeTiling2dInfo {
                dim_m: tensor.shape(tensor.rank() - UInt::new(2)),
                dim_k: tensor.shape(tensor.rank() - UInt::new(1)),
                dim_n: UInt::new(0),
                lhs_stride: tensor.stride(tensor.rank() - UInt::new(2)),
                rhs_stride: UInt::new(0),
                out_stride: UInt::new(0),
            };

            load_lhs_transposed(tensor, coordinates, k, offset, shared_memory, config, info);
        } else {
            let info = CubeTiling2dInfo {
                dim_m: UInt::new(0),
                dim_k: tensor.shape(tensor.rank() - UInt::new(2)),
                dim_n: tensor.shape(tensor.rank() - UInt::new(1)),
                lhs_stride: UInt::new(0),
                rhs_stride: tensor.stride(tensor.rank() - UInt::new(2)),
                out_stride: UInt::new(0),
            };

            load_rhs_plain(tensor, coordinates, k, offset, shared_memory, config, info);
        }

        for i in range(0u32, Comptime::get(sm_size), Comptime::new(false)) {
            sm_out[i] = shared_memory[i];
        }
    }

    /// Exported test
    pub fn load_lhs_transposed_unit_test<R: JitRuntime>(device: &R::Device) {
        let lhs = range_tensor::<R>(16, 16, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(16, 16, 8);

        load_tensor_test_launch::<F32, R>(
            lhs.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(TILE_SIZE as u8, &lhs.handle, &lhs.strides, &lhs.shape.dims),
            ArrayArg::vectorized(TILE_SIZE as u8, &sm_out, 64),
            4,
            4,
            8,
            config,
            true,
        );

        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 76.0, 92.0, 108.0, 124.0, 0.0, 0.0, 0.0, 0.0, 77.0, 93.0, 109.0, 125.0, 0.0,
            0.0, 0.0, 0.0, 78.0, 94.0, 110.0, 126.0, 0.0, 0.0, 0.0, 0.0, 79.0, 95.0, 111.0, 127.0,
        ];
        assert_equals::<R>(sm_out, expected, device);
    }

    /// Exported test
    pub fn load_lhs_transposed_out_of_bounds_cube_test<R: JitRuntime>(device: &R::Device) {
        let vectorization_factor = 1;
        let lhs = range_tensor::<R>(5, 1, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(2, 2, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(5, 1, 1);

        load_tensor_multiple_tiles_test_launch::<F32, R>(
            lhs.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                vectorization_factor as u8,
                &lhs.handle,
                &lhs.strides,
                &lhs.shape.dims,
            ),
            ArrayArg::vectorized(TILE_SIZE as u8, &sm_out, 64),
            0,
            config,
            true,
        );

        let expected = &[
            0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_equals::<R>(sm_out, expected, device);
    }

    /// Exported test
    pub fn load_lhs_transposed_cube_test<R: JitRuntime>(device: &R::Device) {
        let lhs = range_tensor::<R>(8, 8, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(2, 2, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(8, 8, 8);

        load_tensor_multiple_tiles_test_launch::<F32, R>(
            lhs.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(TILE_SIZE as u8, &lhs.handle, &lhs.strides, &lhs.shape.dims),
            ArrayArg::vectorized(TILE_SIZE as u8, &sm_out, 64),
            0,
            config,
            true,
        );

        let expected = &[
            0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 1.0, 9.0, 17.0, 25.0, 33.0, 41.0, 49.0,
            57.0, 2.0, 10.0, 18.0, 26.0, 34.0, 42.0, 50.0, 58.0, 3.0, 11.0, 19.0, 27.0, 35.0, 43.0,
            51.0, 59.0, 4.0, 12.0, 20.0, 28.0, 36.0, 44.0, 52.0, 60.0, 5.0, 13.0, 21.0, 29.0, 37.0,
            45.0, 53.0, 61.0, 6.0, 14.0, 22.0, 30.0, 38.0, 46.0, 54.0, 62.0, 7.0, 15.0, 23.0, 31.0,
            39.0, 47.0, 55.0, 63.0,
        ];
        assert_equals::<R>(sm_out, expected, device);
    }

    /// Exported test
    pub fn load_lhs_transposed_offset_cube_test<R: JitRuntime>(device: &R::Device) {
        let lhs = range_tensor::<R>(8, 16, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(2, 2, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(8, 8, 16);

        load_tensor_multiple_tiles_test_launch::<F32, R>(
            lhs.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(TILE_SIZE as u8, &lhs.handle, &lhs.strides, &lhs.shape.dims),
            ArrayArg::vectorized(TILE_SIZE as u8, &sm_out, 64),
            8,
            config,
            true,
        );

        let expected = &[
            8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 9.0, 25.0, 41.0, 57.0, 73.0, 89.0,
            105.0, 121.0, 10.0, 26.0, 42.0, 58.0, 74.0, 90.0, 106.0, 122.0, 11.0, 27.0, 43.0, 59.0,
            75.0, 91.0, 107.0, 123.0, 12.0, 28.0, 44.0, 60.0, 76.0, 92.0, 108.0, 124.0, 13.0, 29.0,
            45.0, 61.0, 77.0, 93.0, 109.0, 125.0, 14.0, 30.0, 46.0, 62.0, 78.0, 94.0, 110.0, 126.0,
            15.0, 31.0, 47.0, 63.0, 79.0, 95.0, 111.0, 127.0,
        ];
        assert_equals::<R>(sm_out, expected, device);
    }

    /// Exported test
    pub fn load_rhs_plain_unit_test<R: JitRuntime>(device: &R::Device) {
        let rhs = range_tensor::<R>(16, 16, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(8, 16, 16);

        load_tensor_test_launch::<F32, R>(
            rhs.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(TILE_SIZE as u8, &rhs.handle, &rhs.strides, &rhs.shape.dims),
            ArrayArg::vectorized(TILE_SIZE as u8, &sm_out, 64),
            4,
            4,
            8,
            config,
            false,
        );

        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 196.0, 197.0, 198.0, 199.0, 0.0, 0.0, 0.0, 0.0, 212.0, 213.0, 214.0, 215.0,
            0.0, 0.0, 0.0, 0.0, 228.0, 229.0, 230.0, 231.0, 0.0, 0.0, 0.0, 0.0, 244.0, 245.0,
            246.0, 247.0,
        ];
        assert_equals::<R>(sm_out, expected, device);
    }

    /// Exported test
    pub fn load_rhs_plain_cube_test<R: JitRuntime>(device: &R::Device) {
        let rhs = range_tensor::<R>(8, 8, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(2, 2, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(8, 8, 8);

        load_tensor_multiple_tiles_test_launch::<F32, R>(
            rhs.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(TILE_SIZE as u8, &rhs.handle, &rhs.strides, &rhs.shape.dims),
            ArrayArg::vectorized(TILE_SIZE as u8, &sm_out, 64),
            0,
            config,
            false,
        );

        let expected = &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
            44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
            58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
        ];
        assert_equals::<R>(sm_out, expected, device);
    }

    /// Exported test
    pub fn load_rhs_plain_cube_offset_test<R: JitRuntime>(device: &R::Device) {
        let rhs = range_tensor::<R>(16, 8, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(2, 2, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(16, 16, 8);

        load_tensor_multiple_tiles_test_launch::<F32, R>(
            rhs.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(TILE_SIZE as u8, &rhs.handle, &rhs.strides, &rhs.shape.dims),
            ArrayArg::vectorized(TILE_SIZE as u8, &sm_out, 64),
            8,
            config,
            false,
        );

        let expected = &[
            64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0,
            78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0,
            92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
            105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0,
            117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0,
        ];
        assert_equals::<R>(sm_out, expected, device);
    }
}
