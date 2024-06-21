use burn_cube::prelude::*;

use crate::{kernel::matmul::Tiling2dConfig, JitBackend, JitRuntime};

use super::config::CubeTiling2dConfig;

#[cube]
fn load_tile<F: Float>(
    tensor: Tensor<F>,
    tile: Array<F>,
    cube_offset: UInt,
    load_row: UInt,
    load_col: UInt,
    skip_row: UInt,
    skip_col: UInt,
    check_vertical_bounds: Comptime<bool>,
    check_horizontal_bounds: Comptime<bool>,
    config: Comptime<CubeTiling2dConfig>,
) -> Array<F> {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll);

    let rank = tensor.rank();
    let tensor_stride = tensor.stride(rank - UInt::new(2)) / Comptime::runtime(tile_size);
    let tensor_position_base =
        load_row * tensor_stride + load_col / Comptime::runtime(tile_size) + cube_offset;

    if Comptime::get(check_vertical_bounds) {
        let row = skip_row + load_row;
        let dim_vertical = tensor.shape(tensor.rank() - UInt::new(2));

        if Comptime::get(check_horizontal_bounds) {
            let col = skip_col + load_col;
            let dim_horizontal = tensor.shape(tensor.rank() - UInt::new(1));

            if col >= dim_horizontal {
                read_zeros(tile, tile_size, unroll);
            } else {
                read_partial(
                    tensor,
                    dim_vertical,
                    row,
                    tensor_position_base,
                    tensor_stride,
                    tile,
                    tile_size,
                );
            }
        } else {
            read_partial(
                tensor,
                dim_vertical,
                row,
                tensor_position_base,
                tensor_stride,
                tile,
                tile_size,
            );
        }
    } else {
        if Comptime::get(check_horizontal_bounds) {
            let col = skip_col + load_col;
            let dim_horizontal = tensor.shape(tensor.rank() - UInt::new(1));
            if col >= dim_horizontal {
                read_zeros(tile, tile_size, unroll);
            } else {
                read_whole(
                    tensor,
                    tensor_position_base,
                    tensor_stride,
                    tile,
                    tile_size,
                    unroll,
                );
            }
        } else {
            read_whole(
                tensor,
                tensor_position_base,
                tensor_stride,
                tile,
                tile_size,
                unroll,
            );
        }
    }

    tile
}

#[cube]
fn write_tile_plain<F: Float>(
    tile: Array<F>,
    mut shared_memory: SharedMemory<F>,
    sm_position_base: UInt,
    sm_stride: UInt,
    unroll: Comptime<bool>,
    tile_size: Comptime<UInt>,
) {
    for i in range(0u32, Comptime::get(tile_size), unroll) {
        shared_memory[sm_position_base + i * sm_stride] = tile[i];
    }
}

#[cube]
fn write_tile_transposed<F: Float>(
    tile: Array<F>,
    mut shared_memory: SharedMemory<F>,
    sm_position_base: UInt,
    sm_stride: UInt,
    unroll: Comptime<bool>,
    tile_size: Comptime<UInt>,
) {
    let is_scalar = Comptime::map(tile_size, |c| c.val == 1);
    if Comptime::get(is_scalar) {
        shared_memory[sm_position_base] = tile[0];
    } else {
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let mut transposed = Array::<F>::new(Comptime::get(tile_size));
            for j in range(0u32, Comptime::get(tile_size), unroll) {
                transposed[j] = tile[j][i];
            }
            let sm_position = sm_position_base + i * sm_stride;
            shared_memory[sm_position] = transposed.to_vectorized(tile_size);
        }
    }
}

#[cube]
fn load_lhs_transposed<F: Float>(
    lhs: Tensor<F>,
    unit_row: UInt,
    unit_col: UInt,
    skip_row: UInt,
    skip_col: UInt,
    k: UInt,
    offset_lhs: UInt,
    shared_lhs: SharedMemory<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let tile_size = Comptime::map(config, |c| c.tile_size);

    let sm_stride = Comptime::runtime(block_size_m / tile_size);

    let sm_position_base = unit_col * sm_stride + unit_row / Comptime::runtime(tile_size);
    let cube_offset = offset_lhs + k / Comptime::runtime(tile_size);

    let tile = Array::<F>::vectorized(Comptime::get(tile_size), Comptime::get(tile_size));

    load_tile(
        lhs,
        tile,
        cube_offset,
        unit_col,
        unit_row,
        skip_row,
        skip_col,
        Comptime::map(config, |c| c.check_m_bounds),
        Comptime::map(config, |c| c.check_k_bounds),
        config,
    );

    write_tile_transposed(
        tile,
        shared_lhs,
        sm_position_base,
        sm_stride,
        Comptime::map(config, |c| c.unroll),
        tile_size,
    );
}

#[cube]
fn load_rhs_plain<F: Float>(
    rhs: Tensor<F>,
    unit_row: UInt,
    unit_col: UInt,
    skip_row: UInt,
    skip_col: UInt,
    k: UInt,
    offset_rhs: UInt,
    shared_rhs: SharedMemory<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let tile_size = Comptime::map(config, |c| c.tile_size);

    let sm_stride = Comptime::runtime(block_size_n / tile_size);
    let tensor_stride = rhs.stride(rhs.rank() - UInt::new(2)) / Comptime::runtime(tile_size);

    let sm_position_base = unit_row * sm_stride + unit_col / Comptime::runtime(tile_size);
    let cube_offset = offset_rhs + k * tensor_stride;

    let tile = Array::<F>::vectorized(Comptime::get(tile_size), Comptime::get(tile_size));

    load_tile(
        rhs,
        tile,
        cube_offset,
        unit_row,
        unit_col,
        skip_row,
        skip_col,
        Comptime::map(config, |c| c.check_k_bounds),
        Comptime::map(config, |c| c.check_n_bounds),
        config,
    );

    write_tile_plain(
        tile,
        shared_rhs,
        sm_position_base,
        sm_stride,
        Comptime::map(config, |c| c.unroll),
        tile_size,
    );
}

#[cube]
fn read_zeros<F: Float>(mut tile: Array<F>, tile_size: Comptime<UInt>, unroll: Comptime<bool>) {
    let zeros = F::vectorized(0., Comptime::get(tile_size));
    for i in range(0u32, Comptime::get(tile_size), unroll) {
        tile[i] = zeros;
    }
}

#[cube]
fn read_partial<F: Float>(
    tensor: Tensor<F>,
    dim_vertical: UInt,
    row: UInt,
    position_base: UInt,
    stride: UInt,
    mut tile: Array<F>,
    tile_size: Comptime<UInt>,
) {
    let num_reads = UInt::min(dim_vertical - row, Comptime::runtime(tile_size));
    for i in range(0u32, num_reads, Comptime::new(false)) {
        tile[i] = tensor[position_base + i * stride];
    }
    let zeros = F::vectorized(0., Comptime::get(tile_size));
    for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
        tile[i] = zeros;
    }
}

#[cube]
fn read_whole<F: Float>(
    tensor: Tensor<F>,
    position_base: UInt,
    stride: UInt,
    mut tile: Array<F>,
    tile_size: Comptime<UInt>,
    unroll: Comptime<bool>,
) {
    for i in range(0u32, Comptime::get(tile_size), unroll) {
        tile[i] = tensor[position_base + i * stride];
    }
}

// TODO Bug: Impossible to launch if the function has no generics
// TODO mut is not obligatory in rust syntax, but obligatory to be considered output
#[cube(launch)]
fn read_whole_test<F: Float>(tensor: Tensor<F>, mut tile: Array<F>, tile_size: Comptime<UInt>) {
    read_whole(
        tensor,
        UInt::new(0),
        tensor.stride(1),
        tile,
        tile_size,
        Comptime::new(true),
    )
}

#[cube(launch)]
fn read_partial_test<F: Float>(tensor: Tensor<F>, mut tile: Array<F>, tile_size: Comptime<UInt>) {
    read_partial(
        tensor,
        Comptime::runtime(tile_size),
        UInt::new(2),
        UInt::new(2),
        tensor.stride(1),
        tile,
        tile_size,
    )
}

#[cube(launch)]
fn read_zeros_test<F: Float>(mut tile: Array<F>, tile_size: Comptime<UInt>) {
    read_zeros(tile, tile_size, Comptime::new(true))
}

#[cube(launch)]
fn load_tile_test<F: Float>(
    lhs: Tensor<F>,
    mut tile: Array<F>,
    unit_row: UInt,
    unit_col: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let cube_offset = UInt::new(0);
    let check_vertical_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_horizontal_bounds = Comptime::map(config, |c| c.check_k_bounds);

    load_tile::<F>(
        lhs,
        tile,
        cube_offset,
        unit_row,
        unit_col,
        UInt::new(0),
        UInt::new(0),
        check_vertical_bounds,
        check_horizontal_bounds,
        config,
    );
}

#[cube(launch)]
fn write_tile_test<F: Float>(
    tile: Array<F>,
    mut sm_out: Array<F>,
    config: Comptime<CubeTiling2dConfig>,
    transposed: Comptime<bool>,
) {
    let unroll = Comptime::map(config, |c| c.unroll);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::map(config, |c| c.block_size_k);

    let sm_stride = block_size_m / tile_size;
    let sm_size = Comptime::runtime(block_size_k * sm_stride);
    let shared_memory = SharedMemory::vectorized(sm_size, Comptime::get(tile_size));

    if Comptime::get(transposed) {
        write_tile_transposed(
            tile,
            shared_memory,
            UInt::new(0),
            Comptime::runtime(sm_stride),
            unroll,
            tile_size,
        );
    } else {
        write_tile_plain(
            tile,
            shared_memory,
            UInt::new(0),
            Comptime::runtime(sm_stride),
            unroll,
            tile_size,
        );
    }

    for i in range(0u32, sm_size, Comptime::new(false)) {
        sm_out[i] = shared_memory[i];
    }
}

#[cube(launch)]
fn load_tensor_test<F: Float>(
    tensor: Tensor<F>,
    mut sm_out: Array<F>,
    config: Comptime<CubeTiling2dConfig>,
    is_lhs: Comptime<bool>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let sm_size = block_size_k * block_size_m;
    let shared_memory =
        SharedMemory::<F>::vectorized(Comptime::get(sm_size), Comptime::get(tile_size));

    let unit_row = UInt::new(4);
    let unit_col = UInt::new(4);
    let k = UInt::new(8);
    let offset = UInt::new(0);

    if Comptime::get(is_lhs) {
        load_lhs_transposed(
            tensor,
            unit_row,
            unit_col,
            UInt::new(0),
            UInt::new(0),
            k,
            offset,
            shared_memory,
            config,
        );
    } else {
        load_rhs_plain(
            tensor,
            unit_row,
            unit_col,
            UInt::new(0),
            UInt::new(0),
            k,
            offset,
            shared_memory,
            config,
        );
    }

    for i in range(0u32, Comptime::get(sm_size), Comptime::new(false)) {
        sm_out[i] = shared_memory[i];
    }
}

pub fn read_whole_unit_test<R: JitRuntime>(device: &R::Device) {
    pub type B<R> = JitBackend<R, f32, i32>;

    let tile_size = 4;
    let tensor = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
        .reshape([4, 4])
        .float()
        .into_primitive();
    let client = R::client(device);

    let tile = client.empty(tile_size * tile_size * core::mem::size_of::<f32>());

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_output(0, tile_size as u8);

    read_whole_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
        ArrayHandle::new(&tile, 4),
        tile_size.into(),
    );

    let actual = client.read(tile.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ];
    assert_eq!(actual, expected);
}

pub fn read_partial_unit_test<R: JitRuntime>(device: &R::Device) {
    pub type B<R> = JitBackend<R, f32, i32>;

    let tile_size = 4;
    let tensor = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
        .reshape([4, 4])
        .float()
        .into_primitive();
    let client = R::client(device);

    let tile = client.empty(tile_size * tile_size * core::mem::size_of::<f32>());

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_output(0, tile_size as u8);

    read_partial_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
        ArrayHandle::new(&tile, 4),
        tile_size.into(),
    );

    let actual = client.read(tile.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(actual, expected);
}

pub fn read_zeros_unit_test<R: JitRuntime>(device: &R::Device) {
    let tile_size = 4;
    let client = R::client(device);

    let tile = client.empty(tile_size * tile_size * core::mem::size_of::<f32>());

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_output(0, tile_size as u8);

    read_zeros_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        ArrayHandle::new(&tile, 4),
        tile_size.into(),
    );

    let actual = client.read(tile.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(actual, expected);
}

pub fn load_tile_no_checks_unit_test<R: JitRuntime>(device: &R::Device) {
    pub type B<R> = JitBackend<R, f32, i32>;

    let tile_size = 4;
    let tensor = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..64, device)
        .reshape([8, 8])
        .float()
        .into_primitive();
    let client = R::client(device);

    let tile = client.empty(tile_size * tile_size * core::mem::size_of::<f32>());

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_output(0, tile_size as u8);

    let mut tiling2d_config = Tiling2dConfig::default();
    tiling2d_config.block_size_m = 8;
    tiling2d_config.block_size_k = 8;
    tiling2d_config.block_size_n = 8;
    let config = CubeTiling2dConfig::new(tiling2d_config, 8, 8, 8, tile_size);

    load_tile_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
        ArrayHandle::new(&tile, 4),
        0,
        0,
        config,
    );

    let actual = client.read(tile.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 16.0, 17.0, 18.0, 19.0, 24.0, 25.0, 26.0, 27.0,
    ];
    assert_eq!(actual, expected);
}

pub fn load_tile_vertical_checks_unit_test<R: JitRuntime>(device: &R::Device) {
    pub type B<R> = JitBackend<R, f32, i32>;

    let tile_size = 4;
    let tensor = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..64, device)
        .reshape([6, 8])
        .float()
        .into_primitive();
    let client = R::client(device);

    let tile = client.empty(tile_size * tile_size * core::mem::size_of::<f32>());

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_output(0, tile_size as u8);

    let mut tiling2d_config = Tiling2dConfig::default();
    tiling2d_config.block_size_m = 8;
    tiling2d_config.block_size_k = 8;
    tiling2d_config.block_size_n = 8;
    let config = CubeTiling2dConfig::new(tiling2d_config, 6, 8, 8, tile_size);

    load_tile_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
        ArrayHandle::new(&tile, 4),
        4,
        0,
        config,
    );

    let actual = client.read(tile.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        32.0, 33.0, 34.0, 35.0, 40.0, 41.0, 42.0, 43.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(actual, expected);
}

pub fn load_tile_horizontal_checks_unit_test<R: JitRuntime>(device: &R::Device) {
    pub type B<R> = JitBackend<R, f32, i32>;

    let tile_size = 4;
    let tensor = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..64, device)
        .reshape([8, 4])
        .float()
        .into_primitive();
    let client = R::client(device);

    let tile = client.empty(tile_size * tile_size * core::mem::size_of::<f32>());

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_output(0, tile_size as u8);

    let mut tiling2d_config = Tiling2dConfig::default();
    tiling2d_config.block_size_m = 8;
    tiling2d_config.block_size_k = 8;
    tiling2d_config.block_size_n = 8;
    let config = CubeTiling2dConfig::new(tiling2d_config, 8, 4, 8, tile_size);

    load_tile_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
        ArrayHandle::new(&tile, 4),
        0,
        4,
        config,
    );

    let actual = client.read(tile.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(actual, expected);
}

pub fn write_tile_plain_unit_test<R: JitRuntime>(device: &R::Device) {
    pub type B<R> = JitBackend<R, f32, i32>;

    let tile_size = 4;
    let tile = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
        .reshape([4, 4])
        .float()
        .into_primitive();
    let client = R::client(device);

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_output(0, tile_size as u8);

    let mut tiling2d_config = Tiling2dConfig::default();
    tiling2d_config.block_size_m = 8;
    tiling2d_config.block_size_k = 8;
    tiling2d_config.block_size_n = 8;
    let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 8, 8, 8, tile_size);

    let sm_out = client.empty(
        tiling2d_config.block_size_k * tiling2d_config.block_size_m * core::mem::size_of::<f32>(),
    );

    write_tile_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        ArrayHandle::new(&tile.handle, 4),
        ArrayHandle::new(&sm_out, 4),
        config,
        false,
    );

    let actual = client.read(sm_out.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0,
        10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(actual, expected);
}

pub fn write_tile_transposed_unit_test<R: JitRuntime>(device: &R::Device) {
    pub type B<R> = JitBackend<R, f32, i32>;

    let tile_size = 4;
    let tile = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
        .reshape([4, 4])
        .float()
        .into_primitive();
    let client = R::client(device);

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_output(0, tile_size as u8);

    let mut tiling2d_config = Tiling2dConfig::default();
    tiling2d_config.block_size_m = 8;
    tiling2d_config.block_size_k = 8;
    tiling2d_config.block_size_n = 8;
    let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 8, 8, 8, tile_size);

    let sm_out = client.empty(
        tiling2d_config.block_size_k * tiling2d_config.block_size_m * core::mem::size_of::<f32>(),
    );

    write_tile_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        ArrayHandle::new(&tile.handle, 4),
        ArrayHandle::new(&sm_out, 64),
        config,
        true,
    );

    let actual = client.read(sm_out.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        0.0, 4.0, 8.0, 12.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 9.0, 13.0, 0.0, 0.0, 0.0, 0.0, 2.0, 6.0,
        10.0, 14.0, 0.0, 0.0, 0.0, 0.0, 3.0, 7.0, 11.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(actual, expected);
}

pub fn load_lhs_transposed_unit_test<R: JitRuntime>(device: &R::Device) {
    pub type B<R> = JitBackend<R, f32, i32>;

    let tile_size = 4;
    let lhs = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..256, device)
        .reshape([16, 16])
        .float()
        .into_primitive();
    let client = R::client(device);

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_output(0, tile_size as u8);

    let mut tiling2d_config = Tiling2dConfig::default();
    tiling2d_config.block_size_m = 8;
    tiling2d_config.block_size_k = 8;
    tiling2d_config.block_size_n = 8;
    let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 16, 16, 8, tile_size);

    let sm_out = client.empty(
        tiling2d_config.block_size_k * tiling2d_config.block_size_m * core::mem::size_of::<f32>(),
    );

    load_tensor_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        TensorHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
        ArrayHandle::new(&sm_out, 64),
        config,
        true,
    );

    let actual = client.read(sm_out.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        76.0, 92.0, 108.0, 124.0, 0.0, 0.0, 0.0, 0.0, 77.0, 93.0, 109.0, 125.0, 0.0, 0.0, 0.0, 0.0,
        78.0, 94.0, 110.0, 126.0, 0.0, 0.0, 0.0, 0.0, 79.0, 95.0, 111.0, 127.0,
    ];
    assert_eq!(actual, expected);
}

pub fn load_rhs_plain_unit_test<R: JitRuntime>(device: &R::Device) {
    pub type B<R> = JitBackend<R, f32, i32>;

    let tile_size = 4;
    let rhs = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..256, device)
        .reshape([16, 16])
        .float()
        .into_primitive();
    let client = R::client(device);

    // Unit test
    let cube_count = CubeCount::new(1, 1, 1);
    let settings = KernelSettings::default()
        .cube_dim(CubeDim::new(1, 1, 1))
        .vectorize_input(0, tile_size as u8)
        .vectorize_output(0, tile_size as u8);

    let mut tiling2d_config = Tiling2dConfig::default();
    tiling2d_config.block_size_m = 8;
    tiling2d_config.block_size_k = 8;
    tiling2d_config.block_size_n = 8;
    let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 8, 16, 16, tile_size);

    let sm_out = client.empty(
        tiling2d_config.block_size_k * tiling2d_config.block_size_m * core::mem::size_of::<f32>(),
    );

    load_tensor_test_launch::<F32, R>(
        client.clone(),
        cube_count,
        settings,
        TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        ArrayHandle::new(&sm_out, 64),
        config,
        false,
    );

    let actual = client.read(sm_out.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    let expected = &[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        196.0, 197.0, 198.0, 199.0, 0.0, 0.0, 0.0, 0.0, 212.0, 213.0, 214.0, 215.0, 0.0, 0.0, 0.0,
        0.0, 228.0, 229.0, 230.0, 231.0, 0.0, 0.0, 0.0, 0.0, 244.0, 245.0, 246.0, 247.0,
    ];
    assert_eq!(actual, expected);
}
