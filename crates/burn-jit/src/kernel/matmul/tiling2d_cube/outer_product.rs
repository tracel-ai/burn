use burn_cube::prelude::*;

use crate::kernel::matmul::Tiling2dConfig;

use super::config::CubeTiling2dConfig;

#[cube]
pub(crate) fn tile_outer_product<F: Float>(
    register_m: F,
    register_n: F,
    mut results: Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll);
    let is_scalar = Comptime::map(tile_size, |c| c.val == 1);
    if Comptime::get(is_scalar) {
        // works
        results[0] = results[0] + register_m * register_n;
        // doesnt work
        results[0] += register_m * register_n;
    } else {
        for res_idx_m in range(0u32, Comptime::get(tile_size), unroll) {
            let res_pos_base = res_idx_m * Comptime::runtime(tile_size);
            for res_idx_n in range(0u32, Comptime::get(tile_size), unroll) {
                let mul = register_m[res_idx_m] * register_n[res_idx_n];
                // results[res_pos_base + res_idx_n] += mul;
                results[res_pos_base + res_idx_n] = results[res_pos_base + res_idx_n] + mul;
            }
        }
    }
}

#[cube(launch)]
pub fn tile_outer_product_test<F: Float>(
    register_m: Array<F>,
    register_n: Array<F>,
    mut results: Array<F>,
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
    tile_outer_product(register_m, register_n, results, config)
}

fn test_case_config(tile_size: usize) -> CubeTiling2dConfig {
    const SOME_DIM: usize = 12;
    let mut tiling2d_config = Tiling2dConfig::default();
    CubeTiling2dConfig::new(tiling2d_config, SOME_DIM, SOME_DIM, SOME_DIM, tile_size)
}

pub fn tile_outer_product_vectorized_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let register_m = client.create(f32::as_bytes(&[0., 1., 2., 3.]));
    let register_n = client.create(f32::as_bytes(&[1., 2., 3., 4.]));
    let results = client.empty(16 * core::mem::size_of::<f32>());

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default().cube_dim(CubeDim::new(1, 1, 1));
    let config = test_case_config(4);

    tile_outer_product_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        ArrayHandle::new(&register_m, 4),
        ArrayHandle::new(&register_n, 4),
        ArrayHandle::new(&results, 16),
        config,
    );

    let actual = client.read(results.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0,
    ];
    assert_eq!(actual, expected);
}

pub fn tile_outer_product_scalar_unit_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let register_m = client.create(f32::as_bytes(&[3.]));
    let register_n = client.create(f32::as_bytes(&[4.]));
    let results = client.empty(core::mem::size_of::<f32>());

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default().cube_dim(CubeDim::new(1, 1, 1));
    let config = test_case_config(1);

    tile_outer_product_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        ArrayHandle::new(&register_m, 1),
        ArrayHandle::new(&register_n, 1),
        ArrayHandle::new(&results, 1),
        config,
    );

    let actual = client.read(results.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[12.];
    assert_eq!(actual, expected);
}
