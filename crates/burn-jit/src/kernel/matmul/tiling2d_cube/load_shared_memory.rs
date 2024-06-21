use burn_cube::prelude::*;

use crate::{JitBackend, JitRuntime};

use super::config::CubeTiling2dConfig;

#[cube]
fn load_tile<F: Float>(
    tensor: Tensor<F>,
    cube_offset: UInt,
    unit_row: UInt,
    unit_col: UInt,
    tensor_stride: UInt,
    dim_vertical: UInt,
    dim_horizontal: UInt,
    check_vertical_bounds: Comptime<bool>,
    check_horizontal_bounds: Comptime<bool>,
    config: Comptime<CubeTiling2dConfig>,
) -> Array<F> {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll);

    let tensor_position_base = unit_row * tensor_stride + unit_col + cube_offset;
    let tile = Array::<F>::vectorized(Comptime::get(tile_size), Comptime::get(tile_size));

    if Comptime::get(check_vertical_bounds) {
        let row = CUBE_POS_X * Comptime::runtime(block_size_m) + unit_row;
        if Comptime::get(check_horizontal_bounds) {
            let col = CUBE_POS_Y * Comptime::runtime(block_size_n) + unit_col;
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
            let col = CUBE_POS_Y * Comptime::runtime(block_size_n) + unit_col;
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
    entries: Array<F>,
    mut shared_memory: SharedMemory<F>,
    sm_position_base: UInt,
    sm_stride: UInt,
    unroll: Comptime<bool>,
    tile_size: Comptime<UInt>,
) {
    for i in range(0u32, Comptime::get(tile_size), unroll) {
        shared_memory[sm_position_base + i * sm_stride] = entries[i];
    }
}

#[cube]
fn write_tile_transposed<F: Float>(
    entries: Array<F>,
    mut shared_memory: SharedMemory<F>,
    sm_position_base: UInt,
    sm_stride: UInt,
    unroll: Comptime<bool>,
    tile_size: Comptime<UInt>,
) {
    let is_scalar = Comptime::map(tile_size, |c| c.val == 1);
    if Comptime::get(is_scalar) {
        shared_memory[sm_position_base] = entries[0];
    } else {
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let mut transposed = Array::<F>::new(Comptime::get(tile_size));
            for j in range(0u32, Comptime::get(tile_size), unroll) {
                transposed[j] = entries[j][i];
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
    k: UInt,
    offset_lhs: UInt,
    shared_lhs: SharedMemory<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let rank = lhs.rank();
    let dim_m = lhs.shape(rank - UInt::new(2));
    let dim_k = lhs.shape(rank - UInt::new(1));

    let tensor_stride = dim_k / Comptime::runtime(tile_size);
    let sm_stride = block_size_m / tile_size;
    let sm_position_base = unit_col * Comptime::runtime(sm_stride) + unit_row;

    let cube_offset = offset_lhs + k * tensor_stride;

    let tile = load_tile(
        lhs,
        cube_offset,
        unit_row,
        unit_col,
        tensor_stride,
        dim_m,
        dim_k,
        Comptime::map(config, |c| c.check_m_bounds),
        Comptime::map(config, |c| c.check_k_bounds),
        config,
    );

    write_tile_transposed(
        tile,
        shared_lhs,
        sm_position_base,
        tensor_stride,
        Comptime::map(config, |c| c.unroll),
        tile_size,
    );
}

#[cube]
fn load_rhs_plain<F: Float>(
    rhs: Tensor<F>,
    unit_row: UInt,
    unit_col: UInt,
    k: UInt,
    offset_rhs: UInt,
    shared_rhs: SharedMemory<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let rank = rhs.rank();
    let dim_k = rhs.shape(rank - UInt::new(2));
    let dim_n = rhs.shape(rank - UInt::new(1));

    let tensor_stride = dim_n / Comptime::runtime(tile_size);
    let sm_stride = block_size_m / tile_size;
    let sm_position_base = unit_col * Comptime::runtime(sm_stride) + unit_row;

    let cube_offset = offset_rhs + k * tensor_stride;

    let tile = load_tile(
        rhs,
        cube_offset,
        unit_row,
        unit_col,
        tensor_stride,
        dim_k,
        dim_n,
        Comptime::map(config, |c| c.check_k_bounds),
        Comptime::map(config, |c| c.check_n_bounds),
        config,
    );

    write_tile_transposed(
        tile,
        shared_rhs,
        sm_position_base,
        tensor_stride,
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
        tensor.shape(0) / Comptime::runtime(tile_size),
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
        tensor.shape(0) / Comptime::runtime(tile_size),
        tile,
        tile_size,
    )
}

#[cube(launch)]
fn read_zeros_test<F: Float>(mut tile: Array<F>, tile_size: Comptime<UInt>) {
    read_zeros(tile, tile_size, Comptime::new(true))
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
