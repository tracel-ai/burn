use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::{
    base::{Coordinates, Dimensions},
    direct::block_check::{
        base::BlockCheck, horizontal_block_check::HorizontalBlockCheck,
        unchecked_block::UncheckedBlockCheck, vertical_block_check::VerticalBlockCheck,
        whole_block_check::WholeBlockCheck,
    },
};

#[cube]
pub(crate) trait OutputWriter<F: Float>: Sync + Send + 'static {
    fn write_output<B: BlockCheck<F>>(
        out: &mut Tensor<F>,
        results: &Array<F>,
        coordinates: Coordinates,
        offset_output: UInt,
        out_stride: UInt,
        dims: Dimensions,
        config: Comptime<CubeTiling2dConfig>,
    );
}

#[cube]
pub(crate) fn write_to_output<F: Float, W: OutputWriter<F>>(
    out: &mut Tensor<F>,
    results: &Array<F>,
    coordinates: Coordinates,
    offset_output: UInt,
    dims: Dimensions,
    config: Comptime<CubeTiling2dConfig>,
) {
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);

    let out_stride = dims.n;

    if Comptime::get(check_m_bounds) {
        if Comptime::get(check_n_bounds) {
            W::write_output::<WholeBlockCheck>(
                out,
                results,
                coordinates,
                offset_output,
                out_stride,
                dims,
                config,
            );
        } else {
            W::write_output::<VerticalBlockCheck>(
                out,
                results,
                coordinates,
                offset_output,
                out_stride,
                dims,
                config,
            );
        }
    } else {
        if Comptime::get(check_n_bounds) {
            W::write_output::<HorizontalBlockCheck>(
                out,
                results,
                coordinates,
                offset_output,
                out_stride,
                dims,
                config,
            );
        } else {
            W::write_output::<UncheckedBlockCheck>(
                out,
                results,
                coordinates,
                offset_output,
                out_stride,
                dims,
                config,
            );
        }
    }
}

#[cfg(feature = "export_tests")]
/// Exported tests for write output
pub mod tests {
    use crate::{
        kernel::matmul::tiling2d_cube::{
            direct::writer::DirectWriter,
            test_utils::{
                assert_equals, make_config, range_tensor, range_tensor_transposed, zeros_tensor,
                TILE_SIZE,
            },
        },
        JitRuntime,
    };

    use super::{
        super::base::{CoordinatesExpand, DimensionsExpand},
        *,
    };

    #[cube(launch)]
    fn write_to_output_test<F: Float>(
        out: &mut Tensor<F>,
        results: &mut Array<F>,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let coordinates = Coordinates {
            unit_row: UInt::new(4),
            unit_col: UInt::new(4),
            skip_row: UInt::new(0),
            skip_col: UInt::new(0),
        };
        let dims = Dimensions {
            m: out.shape(out.rank() - UInt::new(2)),
            k: UInt::new(0),
            n: out.shape(out.rank() - UInt::new(1)),
        };

        write_to_output::<F, DirectWriter<F>>(
            out,
            results,
            coordinates,
            UInt::new(0),
            dims,
            config,
        );
    }

    #[cube(launch)]
    fn write_results_to_output_out_of_bounds_test<F: Float>(
        out: &mut Tensor<F>,
        results: &mut Array<F>,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let coordinates = Coordinates {
            unit_row: UNIT_POS_X * UInt::new(4),
            unit_col: UNIT_POS_Y * UInt::new(4),
            skip_row: UInt::new(0),
            skip_col: UInt::new(0),
        };
        let dims = Dimensions {
            m: out.shape(out.rank() - UInt::new(2)),
            k: UInt::new(0),
            n: out.shape(out.rank() - UInt::new(1)),
        };

        write_to_output::<F, DirectWriter<F>>(
            out,
            results,
            coordinates,
            UInt::new(0),
            dims,
            config,
        );
    }

    /// Exported test
    pub fn write_to_output_over_height_unit_test<R: JitRuntime>(device: &R::Device) {
        let out = zeros_tensor::<R>(6, 8, device);
        let tile = range_tensor::<R>(4, 4, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = make_config(6, 8, 8);

        write_to_output_test_launch::<F32, R>(
            out.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(TILE_SIZE as u8, &out.handle, &out.strides, &out.shape.dims),
            ArrayArg::new(&tile.handle, 16),
            config,
        );

        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0,
        ];
        assert_equals::<R>(out.handle, expected, device);
    }

    /// Exported test
    pub fn write_to_output_over_width_unit_test<R: JitRuntime>(device: &R::Device) {
        let out = zeros_tensor::<R>(8, 4, device);
        let tile = range_tensor::<R>(4, 4, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = make_config(8, 8, 4);

        write_to_output_test_launch::<F32, R>(
            out.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(TILE_SIZE as u8, &out.handle, &out.strides, &out.shape.dims),
            ArrayArg::new(&tile.handle, 16),
            config,
        );

        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_equals::<R>(out.handle, expected, device);
    }

    /// Exported test
    pub fn write_to_output_vectorized_less_than_tile_unit_test<R: JitRuntime>(device: &R::Device) {
        let vectorization = 2;
        let out = zeros_tensor::<R>(8, 8, device);
        let tile = range_tensor::<R>(4, 4, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = make_config(8, 8, 8);

        write_to_output_test_launch::<F32, R>(
            out.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                vectorization as u8,
                &out.handle,
                &out.strides,
                &out.shape.dims,
            ),
            ArrayArg::new(&tile.handle, 16),
            config,
        );

        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0,
        ];
        assert_equals::<R>(out.handle, expected, device);
    }

    /// Exported test
    pub fn write_to_output_scalar_unit_test<R: JitRuntime>(device: &R::Device) {
        let vectorization = 1;
        let out = zeros_tensor::<R>(8, 8, device);
        let tile = range_tensor::<R>(4, 4, device);
        let cube_dim = CubeDim::new(1, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = make_config(8, 8, 8);

        write_to_output_test_launch::<F32, R>(
            out.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(
                vectorization as u8,
                &out.handle,
                &out.strides,
                &out.shape.dims,
            ),
            ArrayArg::new(&tile.handle, 16),
            config,
        );

        let expected = &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 9.0, 10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 12.0, 13.0, 14.0, 15.0,
        ];
        assert_equals::<R>(out.handle, expected, device);
    }

    /// Exported test
    pub fn write_to_output_scalar_out_of_bounds_cube_test<R: JitRuntime>(device: &R::Device) {
        let vectorization = 1;
        let out = zeros_tensor::<R>(5, 1, device);
        let results = range_tensor_transposed::<R>(4, 4, device);
        let cube_dim = CubeDim::new(2, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        let config = make_config(5, 8, 1);

        write_results_to_output_out_of_bounds_test_launch::<F32, R>(
            out.client.clone(),
            cube_count,
            cube_dim,
            TensorArg::vectorized(vectorization, &out.handle, &out.strides, &out.shape.dims),
            ArrayArg::new(&results.handle, 16),
            config,
        );

        let expected = &[0.0, 1.0, 2.0, 3.0, 0.0];
        assert_equals::<R>(out.handle, expected, device);
    }
}
