use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

#[cube]
pub(crate) fn write_tile_plain<F: Float>(
    tile: &Array<F>,
    mut shared_memory: SharedMemory<F>,
    write_row: UInt,
    write_col: UInt,
    sm_stride: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll_tile);
    let check_sm_bounds = Comptime::map(config, |c| c.check_sm_bounds);
    let tile_size_runtime = Comptime::runtime(tile_size);

    let sm_position_base = write_row * sm_stride + write_col;

    if Comptime::get(check_sm_bounds) {
        let sm_dim_vertical = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
        if write_row < sm_dim_vertical {
            for i in range(0u32, Comptime::get(tile_size), unroll) {
                shared_memory[(sm_position_base + i * sm_stride) / tile_size_runtime] = tile[i];
            }
        }
    } else {
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            shared_memory[(sm_position_base + i * sm_stride) / tile_size_runtime] = tile[i];
        }
    }
}

#[cube]
pub(crate) fn write_tile_transposed<F: Float>(
    tile: &Array<F>,
    mut shared_memory: SharedMemory<F>,
    write_row: UInt,
    write_col: UInt,
    sm_stride: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let check_sm_bounds = Comptime::map(config, |c| c.check_sm_bounds);
    let is_scalar = Comptime::map(tile_size, |c| c.val == 1);

    let sm_position_base = write_row * sm_stride + write_col;

    if Comptime::get(is_scalar) {
        if Comptime::get(check_sm_bounds) {
            let sm_dim_vertical = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
            if write_row < sm_dim_vertical {
                shared_memory[sm_position_base] = tile[0];
            }
        } else {
            shared_memory[sm_position_base] = tile[0];
        }
    } else if Comptime::get(check_sm_bounds) {
        let sm_dim_vertical = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
        if write_row < sm_dim_vertical {
            transpose_tile_to_shared_memory::<F>(
                tile,
                shared_memory,
                sm_position_base,
                sm_stride,
                config,
            );
        }
    } else {
        transpose_tile_to_shared_memory::<F>(
            tile,
            shared_memory,
            sm_position_base,
            sm_stride,
            config,
        );
    }
}

#[cube]
fn transpose_tile_to_shared_memory<F: Float>(
    tile: &Array<F>,
    mut shared_memory: SharedMemory<F>,
    sm_position_base: UInt,
    sm_stride: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll_tile);

    for i in range(0u32, Comptime::get(tile_size), unroll) {
        let mut transposed = F::vectorized_empty(Comptime::get(tile_size));

        // Unrolling this one makes the difference
        for j in range(0u32, Comptime::get(tile_size), unroll) {
            transposed[j] = tile[j][i];
        }

        let sm_position = (sm_position_base + i * sm_stride) / Comptime::runtime(tile_size);
        shared_memory[sm_position] = transposed;
    }
}

#[cfg(feature = "export_tests")]
/// Exported tests for writing tiles to shared memory
pub mod tests {
    use crate::kernel::matmul::tiling2d_cube::test_utils::{
        assert_equals, create_empty, make_config, range_tensor, TILE_SIZE,
    };
    use crate::JitRuntime;

    use super::*;

    #[cube(launch)]
    fn write_tile_test<F: Float>(
        tile: &Array<F>,
        sm_out: &mut Array<F>,
        config: Comptime<CubeTiling2dConfig>,
        transposed: Comptime<bool>,
    ) {
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let block_size_m = Comptime::map(config, |c| c.block_size_m);
        let block_size_k = Comptime::map(config, |c| c.block_size_k);

        let sm_stride = block_size_m;
        let sm_size = Comptime::runtime(block_size_k * block_size_m);
        let shared_memory = SharedMemory::<F>::vectorized(sm_size, Comptime::get(tile_size));

        if Comptime::get(transposed) {
            write_tile_transposed(
                tile,
                shared_memory,
                UInt::new(0),
                UInt::new(0),
                Comptime::runtime(sm_stride),
                config,
            );
        } else {
            write_tile_plain(
                tile,
                shared_memory,
                UInt::new(0),
                UInt::new(0),
                Comptime::runtime(sm_stride),
                config,
            );
        }

        for i in range(0u32, sm_size, Comptime::new(false)) {
            sm_out[i] = shared_memory[i];
        }
    }

    /// Exported test
    pub fn write_tile_plain_unit_test<R: JitRuntime>(device: &R::Device) {
        let tile = range_tensor::<R>(4, 4, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = make_config(8, 8, 8);

        write_tile_test_launch::<F32, R>(
            tile.client.clone(),
            cube_count,
            cube_dim,
            ArrayArg::vectorized(TILE_SIZE as u8, &tile.handle, 4),
            ArrayArg::vectorized(TILE_SIZE as u8, &sm_out, 4),
            config,
            false,
        );

        let expected = &[
            0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0, 8.0,
            9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_equals::<R>(sm_out, expected, device);
    }

    /// Exported test
    pub fn write_tile_transposed_unit_test<R: JitRuntime>(device: &R::Device) {
        let tile = range_tensor::<R>(4, 4, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = make_config(8, 8, 8);

        write_tile_test_launch::<F32, R>(
            tile.client.clone(),
            cube_count,
            cube_dim,
            ArrayArg::vectorized(TILE_SIZE as u8, &tile.handle, 4),
            ArrayArg::vectorized(TILE_SIZE as u8, &sm_out, 64),
            config,
            true,
        );

        let expected = &[
            0.0, 4.0, 8.0, 12.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 9.0, 13.0, 0.0, 0.0, 0.0, 0.0, 2.0,
            6.0, 10.0, 14.0, 0.0, 0.0, 0.0, 0.0, 3.0, 7.0, 11.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_equals::<R>(sm_out, expected, device);
    }
}
