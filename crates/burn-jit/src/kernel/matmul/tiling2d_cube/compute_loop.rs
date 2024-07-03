use burn_cube::prelude::*;

use super::{
    base::Coordinates,
    config::CubeTiling2dConfig,
    outer_product::{tile_outer_product, tile_outer_product_expand},
};

#[cube]
#[allow(unused_mut)]
pub(crate) fn compute_loop<F: Float>(
    coordinates: Coordinates,
    shared_lhs: SharedMemory<F>,
    shared_rhs: SharedMemory<F>,
    results: &mut Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let unroll = Comptime::map(config, |c| c.unroll_compute);

    let unit_row = coordinates.unit_row;
    let unit_col = coordinates.unit_col;

    for dot_index in range(0u32, block_size_k, unroll) {
        let register_m = shared_lhs[(unit_row + dot_index * Comptime::runtime(block_size_m))
            / Comptime::runtime(tile_size)];
        let register_n = shared_rhs[(unit_col + dot_index * Comptime::runtime(block_size_n))
            / Comptime::runtime(tile_size)];

        tile_outer_product::<F>(register_m, register_n, results, config);
    }
}

#[cfg(feature = "export_tests")]
/// Compute loop exported tests
pub mod tests {
    use crate::{
        kernel::matmul::tiling2d_cube::{
            base::TILE_SIZE,
            test_utils::{
                assert_equals, create_empty, make_config, range_tensor, range_tensor_transposed,
            },
        },
        JitRuntime,
    };

    use super::{super::base::CoordinatesExpand, *};

    #[cube(launch)]
    fn compute_loop_test<F: Float>(
        lhs: Tensor<F>,
        rhs: Tensor<F>,
        unit_row: UInt,
        unit_col: UInt,
        results: &mut Array<F>,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let block_size_m = Comptime::map(config, |c| c.block_size_m);
        let block_size_k = Comptime::map(config, |c| c.block_size_m);
        let block_size_n = Comptime::map(config, |c| c.block_size_m);
        let sm_size_lhs = block_size_m * block_size_k / tile_size;
        let sm_size_rhs = block_size_n * block_size_k / tile_size;

        // Shared memories are not launchable, so we launch with tensor and convert to shared memory
        let mut shared_lhs =
            SharedMemory::<F>::vectorized(Comptime::get(sm_size_lhs), Comptime::get(tile_size));
        for i in range(0u32, lhs.len(), Comptime::new(false)) {
            shared_lhs[i] = lhs[i];
        }

        let mut shared_rhs =
            SharedMemory::<F>::vectorized(Comptime::get(sm_size_rhs), Comptime::get(tile_size));
        for i in range(0u32, rhs.len(), Comptime::new(false)) {
            shared_rhs[i] = rhs[i];
        }

        for i in range(0u32, 16u32, Comptime::new(false)) {
            results[i] = F::new(0.);
        }

        let coordinates = Coordinates {
            unit_row,
            unit_col,
            skip_row: UInt::new(0),
            skip_col: UInt::new(0),
        };

        compute_loop(coordinates, shared_lhs, shared_rhs, results, config)
    }

    /// Exported test
    pub fn compute_loop_unit_test<R: JitRuntime>(device: &R::Device) {
        let lhs = range_tensor::<R>(8, 8, device);
        let rhs = range_tensor::<R>(8, 8, device);
        let results = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        const SOME_DIM: usize = 12;
        let config = make_config(SOME_DIM, SOME_DIM, SOME_DIM);

        compute_loop_test_launch::<F32, R>(
            lhs.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(TILE_SIZE as u8, &lhs.handle, &lhs.strides, &lhs.shape.dims),
            TensorArg::vectorized(TILE_SIZE as u8, &rhs.handle, &rhs.strides, &rhs.shape.dims),
            0,
            0,
            ArrayArg::new(&results, 1),
            config,
        );

        let expected = &[
            8960.0, 9184.0, 9408.0, 9632.0, 9184.0, 9416.0, 9648.0, 9880.0, 9408.0, 9648.0, 9888.0,
            10128.0, 9632.0, 9880.0, 10128.0, 10376.0,
        ];
        assert_equals::<R>(results, expected, device);
    }

    /// Exported test
    pub fn compute_loop_unit_offset_test<R: JitRuntime>(device: &R::Device) {
        let lhs = range_tensor_transposed::<R>(8, 4, device);
        let rhs = range_tensor::<R>(4, 8, device);
        let results = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(4, 8, 4);

        compute_loop_test_launch::<F32, R>(
            lhs.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(TILE_SIZE as u8, &lhs.handle, &lhs.strides, &lhs.shape.dims),
            TensorArg::vectorized(TILE_SIZE as u8, &rhs.handle, &rhs.strides, &rhs.shape.dims),
            4,
            4,
            ArrayArg::new(&results, 1),
            config,
        );

        let expected = &[
            1160.0, 1230.0, 1300.0, 1370.0, 1416.0, 1502.0, 1588.0, 1674.0, 1672.0, 1774.0, 1876.0,
            1978.0, 1928.0, 2046.0, 2164.0, 2282.0,
        ];
        assert_equals::<R>(results, expected, device);
    }
}
