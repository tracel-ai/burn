use burn_cube::prelude::*;

use super::config::CubeTiling2dConfig;

#[cube]
pub(crate) fn tile_outer_product<F: Float>(
    register_m: F,
    register_n: F,
    results: &mut Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll);
    let is_scalar = Comptime::map(tile_size, |c| c.val == 1);

    if Comptime::get(is_scalar) {
        results[0] += register_m * register_n;
    } else {
        for res_idx_m in range(0u32, Comptime::get(tile_size), unroll) {
            let res_pos_base = res_idx_m * Comptime::runtime(tile_size);
            for res_idx_n in range(0u32, Comptime::get(tile_size), unroll) {
                let mul = register_m[res_idx_m] * register_n[res_idx_n];
                results[res_pos_base + res_idx_n] += mul;
            }
        }
    }
}

#[cfg(feature = "export_tests")]
/// Exported tests for outer product
pub mod tests {
    use crate::{
        kernel::matmul::tiling2d_cube::test_utils::{assert_equals, create_empty, make_config},
        JitRuntime,
    };

    use super::*;

    #[cube(launch)]
    #[allow(unused_mut)]
    fn tile_outer_product_test<F: Float>(
        register_m: Array<F>,
        register_n: Array<F>,
        results: &mut Array<F>,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        // We launch with array then convert to vectorized float,
        // because direct launch of vectorized float is not supported
        let tile_size = Comptime::map(config, |c| c.tile_size);
        let register_m = register_m.to_vectorized(tile_size);
        let register_n = register_n.to_vectorized(tile_size);

        for i in range(
            0u32,
            Comptime::get(tile_size * tile_size),
            Comptime::new(false),
        ) {
            results[i] = F::new(0.);
        }
        tile_outer_product::<F>(register_m, register_n, results, config)
    }

    /// Exported test
    pub fn tile_outer_product_vectorized_unit_test<R: JitRuntime>(device: &R::Device) {
        let client = R::client(device);
        let register_m = client.create(f32::as_bytes(&[0., 1., 2., 3.]));
        let register_n = client.create(f32::as_bytes(&[1., 2., 3., 4.]));
        let results = create_empty::<R>(4, 4, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        const SOME_DIM: usize = 12;
        let config = make_config(SOME_DIM, SOME_DIM, SOME_DIM);

        tile_outer_product_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            cube_dim,
            ArrayArg::new(&register_m, 4),
            ArrayArg::new(&register_n, 4),
            ArrayArg::new(&results, 16),
            config,
        );

        let expected = &[
            0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0,
        ];
        assert_equals::<R>(results, expected, device);
    }

    /// Exported test
    pub fn tile_outer_product_vectorized_unit_test_2<R: JitRuntime>(device: &R::Device) {
        let client = R::client(device);

        let register_m = client.create(f32::as_bytes(&[16., 20., 24., 28.]));
        let register_n = client.create(f32::as_bytes(&[4., 5., 6., 7.]));
        let results = create_empty::<R>(4, 4, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        const SOME_DIM: usize = 12;
        let config = make_config(SOME_DIM, SOME_DIM, SOME_DIM);

        tile_outer_product_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            cube_dim,
            ArrayArg::new(&register_m, 4),
            ArrayArg::new(&register_n, 4),
            ArrayArg::new(&results, 16),
            config,
        );

        let expected = &[
            64.0, 80.0, 96.0, 112.0, 80.0, 100.0, 120.0, 140.0, 96.0, 120.0, 144.0, 168.0, 112.0,
            140.0, 168.0, 196.0,
        ];
        assert_equals::<R>(results, expected, device);
    }

    /// Exported test
    pub fn tile_outer_product_scalar_unit_test<R: JitRuntime>(device: &R::Device) {
        let client = R::client(device);

        let register_m = client.create(f32::as_bytes(&[3.]));
        let register_n = client.create(f32::as_bytes(&[4.]));
        let results = create_empty::<R>(1, 1, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        const SOME_DIM: usize = 12;
        let config = make_config(SOME_DIM, SOME_DIM, SOME_DIM);

        tile_outer_product_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            cube_dim,
            ArrayArg::new(&register_m, 1),
            ArrayArg::new(&register_n, 1),
            ArrayArg::new(&results, 1),
            config,
        );

        let expected = &[12.];
        assert_equals::<R>(results, expected, device);
    }
}
