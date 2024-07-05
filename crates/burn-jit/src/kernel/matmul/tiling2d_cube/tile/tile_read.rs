use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

#[cube]
pub(crate) fn read_tile_from_global_memory<F: Float>(
    tensor: &Tensor<F>,
    tile: &mut Array<F>,
    cube_offset: UInt,
    read_row: UInt,
    read_col: UInt,
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
    let unroll = Comptime::map(config, |c| c.unroll_tile);

    let tensor_position_base = read_row * tensor_stride + read_col + cube_offset;

    if Comptime::get(check_vertical_bounds) {
        let row = skip_row + read_row;

        if Comptime::get(check_horizontal_bounds) {
            let col = skip_col + read_col;
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
    } else if Comptime::get(check_horizontal_bounds) {
        let col = skip_col + read_col;
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
        let tile_entry = F::vectorized_empty(Comptime::get(tile_size));

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
        let tile_entry = F::vectorized_empty(Comptime::get(tile_size));

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
/// Exported tests for reading tiles in global memory
pub mod tests {
    use crate::kernel::matmul::tiling2d_cube::test_utils::{
        assert_equals, create_empty, make_config, range_tensor, TILE_SIZE,
    };
    use crate::JitRuntime;

    use super::*;

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
    fn read_tile_test<F: Float>(
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

        read_tile_from_global_memory::<F>(
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

    /// Exported test
    pub fn read_whole_vectorized_like_tile_test<R: JitRuntime>(device: &R::Device) {
        let tensor = range_tensor::<R>(4, 4, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        read_whole_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                TILE_SIZE as u8,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            ArrayArg::vectorized(TILE_SIZE as u8, &tile, 4),
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

        read_whole_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                vectorization_factor,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            ArrayArg::vectorized(TILE_SIZE as u8, &tile, 4),
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

        read_whole_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                vectorization_factor,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            ArrayArg::vectorized(TILE_SIZE as u8, &tile, 4),
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

        read_whole_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                vectorization_factor,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            ArrayArg::vectorized(TILE_SIZE as u8, &tile, 4),
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

        read_partial_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                TILE_SIZE as u8,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            ArrayArg::vectorized(TILE_SIZE as u8, &tile, 4),
            TILE_SIZE.into(),
            false,
        );

        let expected = &[
            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_equals::<R>(tile, expected, device);
    }

    /// Exported test
    pub fn read_tile_no_checks_unit_test<R: JitRuntime>(device: &R::Device) {
        let tensor = range_tensor::<R>(8, 8, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(8, 8, 8);

        read_tile_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                TILE_SIZE as u8,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            ArrayArg::vectorized(TILE_SIZE as u8, &tile, 4),
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
    pub fn read_tile_vertical_checks_unit_test<R: JitRuntime>(device: &R::Device) {
        let tensor = range_tensor::<R>(6, 8, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(6, 8, 8);

        read_tile_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                TILE_SIZE as u8,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            ArrayArg::vectorized(TILE_SIZE as u8, &tile, 4),
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
    pub fn read_tile_horizontal_checks_unit_test<R: JitRuntime>(device: &R::Device) {
        let tensor = range_tensor::<R>(8, 4, device);
        let tile = create_empty::<R>(TILE_SIZE, TILE_SIZE, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::new(1, 1, 1);

        let config = make_config(8, 4, 8);

        read_tile_test_launch::<F32, R>(
            tensor.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                TILE_SIZE as u8,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            ArrayArg::vectorized(TILE_SIZE as u8, &tile, 4),
            0,
            4,
            config,
        );

        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_equals::<R>(tile, expected, device);
    }
}
