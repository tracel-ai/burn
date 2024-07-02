use burn_cube::prelude::*;

use super::{
    base::{Coordinates, CubeTiling2dInfo},
    config::CubeTiling2dConfig,
};

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
    let sm_position_base = coordinates.unit_col * sm_stride + coordinates.unit_row;

    let cube_offset = coordinates.skip_row * info.lhs_stride;
    let offset = cube_offset + k + batch_offset;

    let mut tile = Array::<F>::vectorized(Comptime::get(tile_size), Comptime::get(tile_size));

    load_tile::<F>(
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
        sm_position_base,
        sm_stride,
        Comptime::map(config, |c| c.unroll),
        tile_size,
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

    let sm_position_base = coordinates.unit_row * sm_stride + coordinates.unit_col;

    let offset = coordinates.skip_col + k * info.rhs_stride + batch_offset;

    let mut tile = Array::<F>::vectorized(Comptime::get(tile_size), Comptime::get(tile_size));

    load_tile::<F>(
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
        sm_position_base,
        sm_stride,
        Comptime::map(config, |c| c.unroll),
        tile_size,
    );
}

#[cube]
fn load_tile<F: Float>(
    tensor: &Tensor<F>,
    tile: &mut Array<F>,
    cube_offset: UInt,
    load_row: UInt,
    load_col: UInt,
    skip_row: UInt,
    skip_col: UInt,
    tensor_stride: UInt,
    dim_vertical: UInt,
    dim_horizontal: UInt,
    check_vertical_bounds: Comptime<bool>,
    check_horizontal_bounds: Comptime<bool>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll);

    let tensor_position_base = load_row * tensor_stride + load_col + cube_offset;

    if Comptime::get(check_vertical_bounds) {
        let row = skip_row + load_row;

        if Comptime::get(check_horizontal_bounds) {
            let col = skip_col + load_col;
            read_with_both_checks::<F>(
                tensor,
                row,
                col,
                tensor_position_base,
                tensor_stride,
                dim_vertical,
                dim_horizontal,
                tile,
                tile_size,
                unroll,
            );
        } else {
            read_with_vertical_checks::<F>(
                tensor,
                row,
                tensor_position_base,
                tensor_stride,
                dim_vertical,
                tile,
                tile_size,
                unroll,
            );
        }
    } else {
        if Comptime::get(check_horizontal_bounds) {
            let col = skip_col + load_col;
            read_with_horizontal_checks::<F>(
                tensor,
                col,
                tensor_position_base,
                tensor_stride,
                dim_horizontal,
                tile,
                tile_size,
                unroll,
            );
        } else {
            read_without_checks::<F>(
                tensor,
                tensor_position_base,
                tensor_stride,
                tile,
                tile_size,
                unroll,
            );
        }
    }
}

#[cube]
fn write_tile_plain<F: Float>(
    tile: &Array<F>,
    mut shared_memory: SharedMemory<F>,
    sm_position_base: UInt,
    sm_stride: UInt,
    unroll: Comptime<bool>,
    tile_size: Comptime<UInt>,
) {
    let sm_vectorization = Comptime::runtime(tile_size);

    for i in range(0u32, Comptime::get(tile_size), unroll) {
        shared_memory[(sm_position_base + i * sm_stride) / sm_vectorization] = tile[i];
    }
}

#[cube]
fn write_tile_transposed<F: Float>(
    tile: &Array<F>,
    mut shared_memory: SharedMemory<F>,
    sm_position_base: UInt,
    sm_stride: UInt,
    unroll: Comptime<bool>,
    tile_size: Comptime<UInt>,
) {
    let is_scalar = Comptime::map(tile_size, |c| c.val == 1);
    let sm_vectorization = Comptime::runtime(tile_size);

    if Comptime::get(is_scalar) {
        shared_memory[sm_position_base] = tile[0];
    } else {
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            let mut transposed = F::vectorized(0., Comptime::get(tile_size));

            // Unrolling this one makes the difference
            for j in range(0u32, Comptime::get(tile_size), Comptime::new(true)) {
                transposed[j] = tile[j][i];
            }

            let sm_position = (sm_position_base + i * sm_stride) / sm_vectorization;
            shared_memory[sm_position] = transposed;
        }
    }
}

#[cube]
fn read_with_both_checks<F: Float>(
    tensor: &Tensor<F>,
    row: UInt,
    col: UInt,
    position_base: UInt,
    stride: UInt,
    dim_vertical: UInt,
    dim_horizontal: UInt,
    tile: &mut Array<F>,
    tile_size: Comptime<UInt>,
    unroll: Comptime<bool>,
) {
    let tile_size_runtime = Comptime::runtime(tile_size);

    let mut num_reads = UInt::new(0);
    if dim_vertical > row {
        num_reads = UInt::min(dim_vertical - row, tile_size_runtime);
    }

    for i in range(0u32, num_reads, Comptime::new(false)) {
        read_tile_line_with_checks::<F>(
            tensor,
            col,
            position_base,
            stride,
            dim_horizontal,
            tile,
            i,
            tile_size,
            unroll,
        );
    }

    let zeros = F::vectorized(0., Comptime::get(tile_size));
    for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
        tile[i] = zeros;
    }
}

#[cube]
fn read_with_vertical_checks<F: Float>(
    tensor: &Tensor<F>,
    row: UInt,
    position_base: UInt,
    stride: UInt,
    dim_vertical: UInt,
    tile: &mut Array<F>,
    tile_size: Comptime<UInt>,
    unroll: Comptime<bool>,
) {
    let tile_size_runtime = Comptime::runtime(tile_size);

    let mut num_reads = UInt::new(0);
    if dim_vertical > row {
        num_reads = UInt::min(dim_vertical - row, tile_size_runtime);
    }

    for i in range(0u32, num_reads, Comptime::new(false)) {
        read_tile_line_without_checks::<F>(
            tensor,
            position_base,
            stride,
            tile,
            i,
            tile_size,
            unroll,
        );
    }

    let zeros = F::vectorized(0., Comptime::get(tile_size));
    for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
        tile[i] = zeros;
    }
}

#[cube]
fn read_without_checks<F: Float>(
    tensor: &Tensor<F>,
    position_base: UInt,
    stride: UInt,
    tile: &mut Array<F>,
    tile_size: Comptime<UInt>,
    unroll: Comptime<bool>,
) {
    for i in range(0u32, Comptime::get(tile_size), unroll) {
        read_tile_line_without_checks::<F>(
            tensor,
            position_base,
            stride,
            tile,
            i,
            tile_size,
            unroll,
        );
    }
}

#[cube]
fn read_with_horizontal_checks<F: Float>(
    tensor: &Tensor<F>,
    col: UInt,
    position_base: UInt,
    stride: UInt,
    dim_horizontal: UInt,
    tile: &mut Array<F>,
    tile_size: Comptime<UInt>,
    unroll: Comptime<bool>,
) {
    for i in range(0u32, Comptime::get(tile_size), unroll) {
        read_tile_line_with_checks::<F>(
            tensor,
            col,
            position_base,
            stride,
            dim_horizontal,
            tile,
            i,
            tile_size,
            unroll,
        );
    }
}

#[cube]
fn read_tile_line_with_checks<F: Float>(
    tensor: &Tensor<F>,
    col: UInt,
    position_base: UInt,
    stride: UInt,
    dim_horizontal: UInt,
    tile: &mut Array<F>,
    i: UInt,
    tile_size: Comptime<UInt>,
    unroll: Comptime<bool>,
) {
    let vectorization_factor = Comptime::vectorization(tensor);
    let runtime_vectorization = Comptime::runtime(vectorization_factor);

    let position = position_base + i * stride;

    if tile_size == vectorization_factor {
        if col >= dim_horizontal {
            tile[i] = F::vectorized(0., Comptime::get(tile_size));
        } else {
            tile[i] = tensor[position / runtime_vectorization];
        }
    } else {
        let tile_entry = F::vectorized(0., Comptime::get(tile_size));

        let mut num_loops = UInt::new(0);
        if dim_horizontal > col {
            let num_reads = UInt::min(dim_horizontal - col, Comptime::runtime(tile_size));
            num_loops = num_reads / runtime_vectorization;
        }

        for x in range(0u32, num_loops, Comptime::new(false)) {
            read_within_vector::<F>(
                tensor,
                tile_entry,
                position,
                x,
                vectorization_factor,
                unroll,
            );
        }

        tile[i] = tile_entry;
    }
}

#[cube]
fn read_tile_line_without_checks<F: Float>(
    tensor: &Tensor<F>,
    position_base: UInt,
    stride: UInt,
    tile: &mut Array<F>,
    i: UInt,
    tile_size: Comptime<UInt>,
    unroll: Comptime<bool>,
) {
    let vectorization_factor = Comptime::vectorization(tensor);
    let runtime_vectorization = Comptime::runtime(vectorization_factor);

    let position = position_base + i * stride;

    if tile_size == vectorization_factor {
        tile[i] = tensor[position / runtime_vectorization];
    } else {
        let tile_entry = F::vectorized(0., Comptime::get(tile_size));

        for j in range(
            0u32,
            Comptime::get(tile_size / vectorization_factor),
            unroll,
        ) {
            read_within_vector::<F>(
                tensor,
                tile_entry,
                position,
                j,
                vectorization_factor,
                unroll,
            );
        }

        tile[i] = tile_entry;
    }
}

#[cube]
/// Necessary when vectorization_factor < tile_size
fn read_within_vector<F: Float>(
    tensor: &Tensor<F>,
    mut tile_entry: F,
    position: UInt,
    i: UInt,
    vectorization_factor: Comptime<UInt>,
    unroll: Comptime<bool>,
) {
    let is_scalar = Comptime::map(vectorization_factor, |v| v.val == 1);
    let runtime_vectorization = Comptime::runtime(vectorization_factor);

    if Comptime::get(is_scalar) {
        tile_entry[i] = tensor[position + i];
    } else {
        let intermediate = tensor[position / runtime_vectorization + i];

        for j in range(0u32, Comptime::get(vectorization_factor), unroll) {
            tile_entry[i * runtime_vectorization + j] = intermediate[j];
        }
    }
}

#[cfg(feature = "export_tests")]
/// Exported tests for loading to shared memory
pub mod tests {
    use crate::kernel::matmul::tiling2d_cube::{
        base::TILE_SIZE,
        test_utils::{assert_equals, create_empty, make_config, range_tensor},
    };
    use crate::JitRuntime;

    use super::{super::base::CoordinatesExpand, super::base::CubeTiling2dInfoExpand, *};

    #[cube(launch)]
    #[allow(unused_mut)]
    fn read_whole_test<F: Float>(
        tensor: &Tensor<F>,
        tile: &mut Array<F>,
        tile_size: Comptime<UInt>,
        bound_check_horizontal: Comptime<bool>,
    ) {
        if Comptime::get(bound_check_horizontal) {
            read_with_horizontal_checks::<F>(
                tensor,
                UInt::new(0),
                UInt::new(0),
                tensor.stride(0),
                tensor.shape(1),
                tile,
                tile_size,
                Comptime::new(true),
            );
        } else {
            read_without_checks::<F>(
                tensor,
                UInt::new(0),
                tensor.stride(0),
                tile,
                tile_size,
                Comptime::new(true),
            );
        }
    }

    #[cube(launch)]
    #[allow(unused_mut)]
    fn read_partial_test<F: Float>(
        tensor: &Tensor<F>,
        tile: &mut Array<F>,
        tile_size: Comptime<UInt>,
        bound_check_horizontal: Comptime<bool>,
    ) {
        if Comptime::get(bound_check_horizontal) {
            read_with_both_checks::<F>(
                tensor,
                UInt::new(2),
                UInt::new(8),
                UInt::new(0),
                tensor.stride(0),
                tensor.shape(0),
                tensor.shape(1),
                tile,
                tile_size,
                Comptime::new(true),
            );
        } else {
            read_with_vertical_checks::<F>(
                tensor,
                UInt::new(2),
                UInt::new(8),
                tensor.stride(0),
                tensor.shape(0),
                tile,
                tile_size,
                Comptime::new(true),
            );
        }
    }

    #[cube(launch)]
    #[allow(unused_mut)]
    fn load_tile_test<F: Float>(
        lhs: &Tensor<F>,
        tile: &mut Array<F>,
        unit_row: UInt,
        unit_col: UInt,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let cube_offset = UInt::new(0);
        let check_vertical_bounds = Comptime::map(config, |c| c.check_m_bounds);
        let check_horizontal_bounds = Comptime::map(config, |c| c.check_k_bounds);
        let lhs_stride = lhs.stride(lhs.rank() - UInt::new(2));
        let dim_m = lhs.shape(lhs.rank() - UInt::new(2));
        let dim_k = lhs.shape(lhs.rank() - UInt::new(1));

        load_tile::<F>(
            lhs,
            tile,
            cube_offset,
            unit_row,
            unit_col,
            UInt::new(0),
            UInt::new(0),
            lhs_stride,
            dim_m,
            dim_k,
            check_vertical_bounds,
            check_horizontal_bounds,
            config,
        );
    }

    #[cube(launch)]
    fn write_tile_test<F: Float>(
        tile: &Array<F>,
        sm_out: &mut Array<F>,
        config: Comptime<CubeTiling2dConfig>,
        transposed: Comptime<bool>,
    ) {
        let unroll = Comptime::map(config, |c| c.unroll);
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
    pub fn read_whole_vectorized_like_tile_test<R: JitRuntime>(device: &R::Device) {
        let tensor = range_tensor::<R>(4, 4, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        read_whole_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            ArrayHandle::new(&tile, 4),
            TILE_SIZE.into(),
            false,
        );

        assert_equals::<R>(
            tile,
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            device,
        );
    }

    /// Exported test
    pub fn read_whole_vectorized_less_than_tile_test<R: JitRuntime>(device: &R::Device) {
        let vectorization_factor = 2;
        let tensor = range_tensor::<R>(4, 4, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, vectorization_factor as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        read_whole_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            ArrayHandle::new(&tile, 4),
            TILE_SIZE.into(),
            false,
        );

        assert_equals::<R>(
            tile,
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            device,
        );
    }

    /// Exported test
    pub fn read_whole_scalar_test<R: JitRuntime>(device: &R::Device) {
        let vectorization_factor = 1;
        let tensor = range_tensor::<R>(4, 4, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, vectorization_factor as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        read_whole_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            ArrayHandle::new(&tile, 4),
            TILE_SIZE.into(),
            false,
        );

        assert_equals::<R>(
            tile,
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            device,
        );
    }

    /// Exported test
    pub fn read_whole_scalar_out_of_bound_test<R: JitRuntime>(device: &R::Device) {
        let vectorization_factor = 2;
        let tensor = range_tensor::<R>(4, 2, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, vectorization_factor as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        read_whole_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            ArrayHandle::new(&tile, 4),
            TILE_SIZE.into(),
            true,
        );

        assert_equals::<R>(
            tile,
            &[
                0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0, 7.0, 0.0, 0.0,
            ],
            device,
        );
    }

    /// Exported test
    pub fn read_partial_unit_test<R: JitRuntime>(device: &R::Device) {
        let tensor = range_tensor::<R>(4, 4, device);
        let tile = create_empty::<R>(4, 4, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        read_partial_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            ArrayHandle::new(&tile, 4),
            TILE_SIZE.into(),
            false,
        );

        let expected = &[
            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_equals::<R>(tile, expected, device);
    }

    /// Exported test
    pub fn load_tile_no_checks_unit_test<R: JitRuntime>(device: &R::Device) {
        let tensor = range_tensor::<R>(8, 8, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(8, 8, 8);

        load_tile_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            ArrayHandle::new(&tile, 4),
            0,
            0,
            config,
        );

        let expected = &[
            0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 16.0, 17.0, 18.0, 19.0, 24.0, 25.0, 26.0,
            27.0,
        ];
        assert_equals::<R>(tile, expected, device);
    }

    /// Exported test
    pub fn load_tile_vertical_checks_unit_test<R: JitRuntime>(device: &R::Device) {
        let tensor = range_tensor::<R>(6, 8, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(6, 8, 8);

        load_tile_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            ArrayHandle::new(&tile, 4),
            4,
            0,
            config,
        );

        let expected = &[
            32.0, 33.0, 34.0, 35.0, 40.0, 41.0, 42.0, 43.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_equals::<R>(tile, expected, device);
    }

    /// Exported test
    pub fn load_tile_horizontal_checks_unit_test<R: JitRuntime>(device: &R::Device) {
        let tensor = range_tensor::<R>(8, 4, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(8, 4, 8);

        load_tile_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            ArrayHandle::new(&tile, 4),
            0,
            4,
            config,
        );

        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_equals::<R>(tile, expected, device);
    }

    /// Exported test
    pub fn write_tile_plain_unit_test<R: JitRuntime>(device: &R::Device) {
        let tile = range_tensor::<R>(4, 4, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(8, 8, 8);

        write_tile_test_launch::<F32, R>(
            tile.client.clone(),
            cube_count,
            settings,
            ArrayHandle::new(&tile.handle, 4),
            ArrayHandle::new(&sm_out, 4),
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
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(8, 8, 8);

        write_tile_test_launch::<F32, R>(
            tile.client.clone(),
            cube_count,
            settings,
            ArrayHandle::new(&tile.handle, 4),
            ArrayHandle::new(&sm_out, 64),
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

    /// Exported test
    pub fn load_lhs_transposed_unit_test<R: JitRuntime>(device: &R::Device) {
        let lhs = range_tensor::<R>(16, 16, device);
        let sm_out = create_empty::<R>(8, 8, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(16, 16, 8);

        load_tensor_test_launch::<F32, R>(
            lhs.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            ArrayHandle::new(&sm_out, 64),
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

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, vectorization_factor as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(5, 1, 1);

        load_tensor_multiple_tiles_test_launch::<F32, R>(
            lhs.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            ArrayHandle::new(&sm_out, 64),
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

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(8, 8, 8);

        load_tensor_multiple_tiles_test_launch::<F32, R>(
            lhs.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            ArrayHandle::new(&sm_out, 64),
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

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(8, 8, 16);

        load_tensor_multiple_tiles_test_launch::<F32, R>(
            lhs.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            ArrayHandle::new(&sm_out, 64),
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

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(8, 16, 16);

        load_tensor_test_launch::<F32, R>(
            rhs.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ArrayHandle::new(&sm_out, 64),
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

        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(8, 8, 8);

        load_tensor_multiple_tiles_test_launch::<F32, R>(
            rhs.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ArrayHandle::new(&sm_out, 64),
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
        let settings = KernelSettings::default()
            .cube_dim(cube_dim)
            .vectorize_input(0, TILE_SIZE as u8)
            .vectorize_output(0, TILE_SIZE as u8);

        let config = make_config(16, 16, 8);

        load_tensor_multiple_tiles_test_launch::<F32, R>(
            rhs.client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ArrayHandle::new(&sm_out, 64),
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
