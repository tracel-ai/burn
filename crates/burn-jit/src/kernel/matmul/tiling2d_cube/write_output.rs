use burn_cube::prelude::*;

use crate::{kernel::matmul::Tiling2dConfig, JitBackend, JitRuntime};

use super::{base::Coordinates, config::CubeTiling2dConfig};

#[cube]
pub(crate) fn write_to_output<F: Float>(
    out: Tensor<F>,
    results: Array<F>,
    coordinates: Coordinates,
    offset_output: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);

    let row = coordinates.skip_row + coordinates.unit_row;
    let col = coordinates.skip_col + coordinates.unit_col;

    let rank = out.rank();
    let out_stride_row = out.stride(rank - UInt::new(2));

    if Comptime::get(check_m_bounds) {
        let dim_m = out.shape(rank - UInt::new(2));
        if Comptime::get(check_n_bounds) {
            let dim_n = out.shape(rank - UInt::new(1));
            if row < dim_m && col < dim_n {
                let num_writes = UInt::min(dim_m - row, Comptime::runtime(tile_size));
                write_results_to_output_partial::<F>(
                    out,
                    results,
                    row,
                    col,
                    offset_output,
                    out_stride_row,
                    num_writes,
                    config,
                );
            }
        } else {
            if row < dim_m {
                let num_writes = UInt::min(dim_m - row, Comptime::runtime(tile_size));
                write_results_to_output_partial::<F>(
                    out,
                    results,
                    row,
                    col,
                    offset_output,
                    out_stride_row,
                    num_writes,
                    config,
                );
            }
        }
    } else {
        if Comptime::get(check_n_bounds) {
            let dim_n = out.shape(rank - UInt::new(1));
            if col < dim_n {
                write_results_to_output::<F>(
                    out,
                    results,
                    row,
                    col,
                    offset_output,
                    out_stride_row,
                    config,
                );
            }
        } else {
            write_results_to_output::<F>(
                out,
                results,
                row,
                col,
                offset_output,
                out_stride_row,
                config,
            );
        }
    }
}

#[cube]
fn write_results_to_output<F: Float>(
    mut out: Tensor<F>,
    results: Array<F>,
    row: UInt,
    col: UInt,
    offset_output: UInt,
    out_stride_row: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let sm_is_scalar = Comptime::map(tile_size, |t| t.val == 1);
    let unroll = Comptime::map(config, |c| c.unroll);

    let vectorization_factor = Comptime::runtime(Comptime::vectorization(out));
    if Comptime::get(sm_is_scalar) {
        out[(row * out_stride_row + col + offset_output) / vectorization_factor] = results[0];
    } else {
        for res_idx_m in range(0u32, Comptime::get(tile_size), unroll) {
            write_results_inner_loop::<F>(
                out,
                results,
                res_idx_m,
                row,
                col,
                offset_output,
                out_stride_row,
                config,
            )
        }
    }
}

#[cube]
fn write_results_to_output_partial<F: Float>(
    mut out: Tensor<F>,
    results: Array<F>,
    row: UInt,
    col: UInt,
    batch_offset: UInt,
    out_stride_row: UInt,
    num_writes: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let sm_is_scalar = Comptime::map(tile_size, |t| t.val == 1);

    let vectorization_factor = Comptime::runtime(Comptime::vectorization(out));
    if Comptime::get(sm_is_scalar) {
        out[(row * out_stride_row + col + batch_offset) / vectorization_factor] = results[0];
    } else {
        for res_idx_m in range(0u32, num_writes, Comptime::new(false)) {
            write_results_inner_loop::<F>(
                out,
                results,
                res_idx_m,
                row,
                col,
                batch_offset,
                out_stride_row,
                config,
            )
        }
    }
}

#[cube]
fn write_results_inner_loop<F: Float>(
    mut out: Tensor<F>,
    results: Array<F>,
    res_idx_m: UInt,
    row: UInt,
    col: UInt,
    batch_offset: UInt,
    out_stride_row: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll);
    let vectorization_factor = Comptime::vectorization(out);
    let is_scalar = Comptime::map(vectorization_factor, |v| v.val == 1);

    let results_pos_m = res_idx_m * Comptime::runtime(tile_size);
    let out_position = (row + res_idx_m) * out_stride_row + col + batch_offset;

    for i in range(
        0u32,
        Comptime::get(tile_size / vectorization_factor),
        unroll,
    ) {
        if Comptime::get(is_scalar) {
            out[i + out_position] = results[results_pos_m + i];
        } else {
            let mut output_elem = F::vectorized(0., Comptime::get(vectorization_factor));
            for j in range(0u32, Comptime::get(vectorization_factor), unroll) {
                let index = i * Comptime::runtime(vectorization_factor) + j;
                output_elem[j] = results[results_pos_m + index];
            }

            out[i + out_position / Comptime::runtime(vectorization_factor)] = output_elem;
        }
    }
}

#[cfg(feature = "export_tests")]
/// Exported tests for write output
pub mod tests {
    use super::{super::base::CoordinatesExpand, *};

    #[cube(launch)]
    fn write_results_inner_loop_test<F: Float>(
        out: Tensor<F>,
        results: Array<F>,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let out_stride_row = out.stride(out.rank() - UInt::new(2));
        write_results_inner_loop::<F>(
            out,
            results,
            UInt::new(2),
            UInt::new(4),
            UInt::new(4),
            UInt::new(0),
            out_stride_row,
            config,
        );
    }

    #[cube(launch)]
    fn write_results_to_output_test<F: Float>(
        out: Tensor<F>,
        results: Array<F>,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let out_stride_row = out.stride(out.rank() - UInt::new(2));
        write_results_to_output::<F>(
            out,
            results,
            UInt::new(4),
            UInt::new(4),
            UInt::new(0),
            out_stride_row,
            config,
        );
    }

    #[cube(launch)]
    fn write_results_to_output_partial_test<F: Float>(
        out: Tensor<F>,
        results: Array<F>,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let out_stride_row = out.stride(out.rank() - UInt::new(2));
        write_results_to_output_partial::<F>(
            out,
            results,
            UInt::new(4),
            UInt::new(4),
            UInt::new(0),
            out_stride_row,
            UInt::new(2),
            config,
        );
    }

    #[cube(launch)]
    fn write_to_output_over_height_test<F: Float>(
        out: Tensor<F>,
        results: Array<F>,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let coordinates = Coordinates {
            unit_row: UInt::new(4),
            unit_col: UInt::new(4),
            skip_row: UInt::new(0),
            skip_col: UInt::new(0),
        };
        write_to_output::<F>(out, results, coordinates, UInt::new(0), config);
    }

    #[cube(launch)]
    fn write_to_output_over_width_test<F: Float>(
        out: Tensor<F>,
        results: Array<F>,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let coordinates = Coordinates {
            unit_row: UInt::new(4),
            unit_col: UInt::new(4),
            skip_row: UInt::new(0),
            skip_col: UInt::new(0),
        };
        write_to_output::<F>(out, results, coordinates, UInt::new(0), config);
    }

    /// Exported test
    pub fn write_results_inner_loop_unit_test<R: JitRuntime>(device: &R::Device) {
        pub type B<R> = JitBackend<R, f32, i32>;

        let tile_size = 4;
        let out = burn_tensor::Tensor::<B<R>, 2>::zeros([8, 8], device).into_primitive();
        let client = R::client(device);

        let tile = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
            .reshape([4, 4])
            .float()
            .into_primitive();

        // Unit test
        let cube_count = CubeCount::new(1, 1, 1);
        let settings = KernelSettings::default()
            .cube_dim(CubeDim::new(1, 1, 1))
            .vectorize_input(0, tile_size as u8);

        let mut tiling2d_config = Tiling2dConfig::default();
        tiling2d_config.block_size_m = 8;
        tiling2d_config.block_size_k = 8;
        tiling2d_config.block_size_n = 8;
        let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 8, 8, 8, tile_size);

        write_results_inner_loop_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
            ArrayHandle::new(&tile.handle, 16),
            config,
        );

        let actual = client.read(out.handle.binding()).read_sync().unwrap();
        let actual = f32::from_bytes(&actual);
        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(actual, expected);
    }

    /// Exported test
    pub fn write_results_to_output_unit_test<R: JitRuntime>(device: &R::Device) {
        pub type B<R> = JitBackend<R, f32, i32>;

        let tile_size = 4;
        let out = burn_tensor::Tensor::<B<R>, 2>::zeros([8, 8], device).into_primitive();
        let client = R::client(device);

        let tile = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
            .reshape([4, 4])
            .float()
            .into_primitive();

        // Unit test
        let cube_count = CubeCount::new(1, 1, 1);
        let settings = KernelSettings::default()
            .cube_dim(CubeDim::new(1, 1, 1))
            .vectorize_input(0, tile_size as u8);

        let mut tiling2d_config = Tiling2dConfig::default();
        tiling2d_config.block_size_m = 8;
        tiling2d_config.block_size_k = 8;
        tiling2d_config.block_size_n = 8;
        let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 8, 8, 8, tile_size);

        write_results_to_output_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
            ArrayHandle::new(&tile.handle, 16),
            config,
        );

        let actual = client.read(out.handle.binding()).read_sync().unwrap();
        let actual = f32::from_bytes(&actual);
        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0,
        ];
        assert_eq!(actual, expected);
    }

    /// Exported test
    pub fn write_results_to_output_partial_unit_test<R: JitRuntime>(device: &R::Device) {
        pub type B<R> = JitBackend<R, f32, i32>;

        let tile_size = 4;
        let out = burn_tensor::Tensor::<B<R>, 2>::zeros([6, 8], device).into_primitive();
        let client = R::client(device);

        let tile = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
            .reshape([4, 4])
            .float()
            .into_primitive();

        // Unit test
        let cube_count = CubeCount::new(1, 1, 1);
        let settings = KernelSettings::default()
            .cube_dim(CubeDim::new(1, 1, 1))
            .vectorize_input(0, tile_size as u8);

        let mut tiling2d_config = Tiling2dConfig::default();
        tiling2d_config.block_size_m = 8;
        tiling2d_config.block_size_k = 8;
        tiling2d_config.block_size_n = 8;
        let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 6, 8, 8, tile_size);

        write_results_to_output_partial_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
            ArrayHandle::new(&tile.handle, 16),
            config,
        );

        let actual = client.read(out.handle.binding()).read_sync().unwrap();
        let actual = f32::from_bytes(&actual);
        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0,
        ];
        assert_eq!(actual, expected);
    }

    /// Exported test
    pub fn write_to_output_over_height_unit_test<R: JitRuntime>(device: &R::Device) {
        pub type B<R> = JitBackend<R, f32, i32>;

        let tile_size = 4;
        let out = burn_tensor::Tensor::<B<R>, 2>::zeros([6, 8], device).into_primitive();
        let client = R::client(device);

        let tile = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
            .reshape([4, 4])
            .float()
            .into_primitive();

        // Unit test
        let cube_count = CubeCount::new(1, 1, 1);
        let settings = KernelSettings::default()
            .cube_dim(CubeDim::new(1, 1, 1))
            .vectorize_input(0, tile_size as u8);

        let mut tiling2d_config = Tiling2dConfig::default();
        tiling2d_config.block_size_m = 8;
        tiling2d_config.block_size_k = 8;
        tiling2d_config.block_size_n = 8;
        let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 6, 8, 8, tile_size);

        write_to_output_over_height_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
            ArrayHandle::new(&tile.handle, 16),
            config,
        );

        let actual = client.read(out.handle.binding()).read_sync().unwrap();
        let actual = f32::from_bytes(&actual);
        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0,
        ];
        assert_eq!(actual, expected);
    }

    /// Exported test
    pub fn write_to_output_over_width_unit_test<R: JitRuntime>(device: &R::Device) {
        pub type B<R> = JitBackend<R, f32, i32>;

        let tile_size = 4;
        let out = burn_tensor::Tensor::<B<R>, 2>::zeros([8, 4], device).into_primitive();
        let client = R::client(device);

        let tile = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
            .reshape([4, 4])
            .float()
            .into_primitive();

        // Unit test
        let cube_count = CubeCount::new(1, 1, 1);
        let settings = KernelSettings::default()
            .cube_dim(CubeDim::new(1, 1, 1))
            .vectorize_input(0, tile_size as u8);

        let mut tiling2d_config = Tiling2dConfig::default();
        tiling2d_config.block_size_m = 8;
        tiling2d_config.block_size_k = 8;
        tiling2d_config.block_size_n = 8;
        let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 8, 8, 4, tile_size);

        write_to_output_over_width_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
            ArrayHandle::new(&tile.handle, 16),
            config,
        );

        let actual = client.read(out.handle.binding()).read_sync().unwrap();
        let actual = f32::from_bytes(&actual);
        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(actual, expected);
    }

    /// Exported test
    pub fn write_to_output_vectorized_less_than_tile_unit_test<R: JitRuntime>(device: &R::Device) {
        pub type B<R> = JitBackend<R, f32, i32>;

        let tile_size = 4;
        let vectorization = 2;
        let out = burn_tensor::Tensor::<B<R>, 2>::zeros([8, 8], device).into_primitive();
        let client = R::client(device);

        let tile = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
            .reshape([4, 4])
            .float()
            .into_primitive();

        // Unit test
        let cube_count = CubeCount::new(1, 1, 1);
        let settings = KernelSettings::default()
            .cube_dim(CubeDim::new(1, 1, 1))
            .vectorize_input(0, vectorization as u8);

        let mut tiling2d_config = Tiling2dConfig::default();
        tiling2d_config.block_size_m = 8;
        tiling2d_config.block_size_k = 8;
        tiling2d_config.block_size_n = 8;
        let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 8, 8, 8, tile_size);

        write_results_to_output_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
            ArrayHandle::new(&tile.handle, 16),
            config,
        );

        let actual = client.read(out.handle.binding()).read_sync().unwrap();
        let actual = f32::from_bytes(&actual);
        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0,
        ];
        assert_eq!(actual, expected);
    }

    /// Exported test
    pub fn write_to_output_scalar_unit_test<R: JitRuntime>(device: &R::Device) {
        pub type B<R> = JitBackend<R, f32, i32>;

        let tile_size = 4;
        let vectorization = 1;
        let out = burn_tensor::Tensor::<B<R>, 2>::zeros([8, 8], device).into_primitive();
        let client = R::client(device);

        let tile = burn_tensor::Tensor::<B<R>, 1, burn_tensor::Int>::arange(0..16, device)
            .reshape([4, 4])
            .float()
            .into_primitive();

        // Unit test
        let cube_count = CubeCount::new(1, 1, 1);
        let settings = KernelSettings::default()
            .cube_dim(CubeDim::new(1, 1, 1))
            .vectorize_input(0, vectorization as u8);

        let mut tiling2d_config = Tiling2dConfig::default();
        tiling2d_config.block_size_m = 8;
        tiling2d_config.block_size_k = 8;
        tiling2d_config.block_size_n = 8;
        let config = CubeTiling2dConfig::new(tiling2d_config.clone(), 8, 8, 8, tile_size);

        write_results_to_output_test_launch::<F32, R>(
            client.clone(),
            cube_count,
            settings,
            TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
            ArrayHandle::new(&tile.handle, 16),
            config,
        );

        let actual = client.read(out.handle.binding()).read_sync().unwrap();
        let actual = f32::from_bytes(&actual);
        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0,
        ];
        assert_eq!(actual, expected);
    }
}
