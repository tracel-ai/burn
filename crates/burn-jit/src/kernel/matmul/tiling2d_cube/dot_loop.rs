use burn_cube::prelude::*;
use burn_tensor::backend::Backend;

use crate::{kernel::matmul::Tiling2dConfig, JitBackend, JitRuntime};

use super::{
    config::CubeTiling2dConfig,
    outer_product::{tile_outer_product, tile_outer_product_expand},
};

#[cube]
pub(crate) fn dot_loop<F: Float>(
    unit_row: UInt,
    unit_col: UInt,
    shared_lhs: SharedMemory<F>,
    shared_rhs: SharedMemory<F>,
    results: Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    // TODO Comptime arithmetic
    let tile_size = Comptime::runtime(Comptime::map(config, |c| c.tile_size));
    let block_size_m = Comptime::runtime(Comptime::map(config, |c| c.block_size_m));
    let block_size_k = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
    let block_size_n = Comptime::runtime(Comptime::map(config, |c| c.block_size_n));
    let unroll = Comptime::map(config, |c| c.unroll);

    let lhs_stride = block_size_m / tile_size;
    let rhs_stride = block_size_n / tile_size;

    for dot_index in range(0u32, block_size_k, unroll) {
        let register_m = shared_lhs[(unit_col + dot_index) * lhs_stride];
        let register_n = shared_rhs[(unit_row + dot_index) * rhs_stride];

        tile_outer_product(register_m, register_n, results, config);
    }
}

#[cube(launch)]
pub fn dot_loop_test<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    unit_row: UInt,
    unit_col: UInt,
    mut results: Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);

    // Shared memories are not launchable, so we launch with tensor and convert to shared memory
    let sm_size_lhs = Comptime::map(config, |c| c.sm_size_lhs);
    let mut shared_lhs =
        SharedMemory::<F>::vectorized(Comptime::get(sm_size_lhs), Comptime::get(tile_size));
    for i in range(0u32, lhs.len(), Comptime::new(false)) {
        shared_lhs[i] = lhs[i];
    }

    let sm_size_rhs = Comptime::map(config, |c| c.sm_size_rhs);
    let mut shared_rhs =
        SharedMemory::<F>::vectorized(Comptime::get(sm_size_rhs), Comptime::get(tile_size));
    for i in range(0u32, rhs.len(), Comptime::new(false)) {
        shared_rhs[i] = rhs[i];
    }

    for i in range(0u32, 16u32, Comptime::new(false)) {
        results[i] = F::new(0.);
    }

    dot_loop(unit_row, unit_col, shared_lhs, shared_rhs, results, config)
}

pub fn dot_loop_unit_test<R: JitRuntime>(device: &R::Device) {
    pub type B<R> = JitBackend<R, f32, i32>;

    let tile_size = 4;
    let lhs = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..64, device)
        .reshape([8, 8])
        .float()
        .into_primitive();
    let rhs = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..64, device)
        .reshape([8, 8])
        .float()
        .into_primitive();
    let client = R::client(device);

    let unit_row = 0;
    let unit_col = 0;
    let results = client.empty(tile_size * tile_size * core::mem::size_of::<f32>());

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_input(1, tile_size as u8);

    const SOME_DIM: usize = 12;
    let mut tiling2d_config = Tiling2dConfig::default();
    tiling2d_config.block_size_m = 8;
    tiling2d_config.block_size_k = 8;
    tiling2d_config.block_size_n = 8;
    let config = CubeTiling2dConfig::new(tiling2d_config, SOME_DIM, SOME_DIM, SOME_DIM, tile_size);

    dot_loop_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        TensorHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
        TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        unit_row,
        unit_col,
        ArrayHandle::new(&results, 1),
        config,
    );

    let actual = client.read(results.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        8960.0, 9184.0, 9408.0, 9632.0, 9184.0, 9416.0, 9648.0, 9880.0, 9408.0, 9648.0, 9888.0,
        10128.0, 9632.0, 9880.0, 10128.0, 10376.0,
    ];
    assert_eq!(actual, expected);
}
