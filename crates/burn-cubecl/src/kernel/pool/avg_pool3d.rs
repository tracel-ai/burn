use super::pool3d::{
    Pool3dDirectArgsLaunch, Pool3dDirectStrategy, Pool3dDirectStrategyFamily, pool3d_direct,
};
use crate::{
    CubeRuntime,
    kernel::into_contiguous,
    ops::{
        max_line_size, numeric::empty_device_dtype, permute_ncdhw_to_ndhwc, permute_ndhwc_to_ncdhw,
    },
    tensor::CubeTensor,
};
use burn_backend::{Shape, ops::conv::calculate_pool_output_size};
use cubecl::prelude::*;
use cubecl::{CubeDim, calculate_cube_count_elemwise, prelude::ScalarArg};

struct AvgPool3dStrategy;

impl Pool3dDirectStrategyFamily for AvgPool3dStrategy {
    type Indices = ();
    type Config = AvgPool3dStrategyConfig;
    type Pool3d<N: Numeric> = Self;
}

#[derive(CubeType, Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct AvgPool3dStrategyConfig {
    count_include_pad: bool,
    /// Total padded depth (input_depth + 2 * padding_0)
    padded_d: u32,
    /// Total padded height (input_height + 2 * padding_1)
    padded_h: u32,
    /// Total padded width (input_width + 2 * padding_2)
    padded_w: u32,
}

#[cube]
impl<N: Numeric> Pool3dDirectStrategy<N> for AvgPool3dStrategy {
    type Accumulator = (Line<N>, u32);
    type Config = AvgPool3dStrategyConfig;
    type Indices = ();

    fn initialize(
        #[comptime] _config: &Self::Config,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        let sum = Line::empty(line_size).fill(N::from_int(0));
        // Count will be set dynamically: either by accumulate (count_include_pad=false)
        // or by count_position (count_include_pad=true)
        let count = 0u32;

        (sum, count)
    }

    fn accumulate(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        _index: u32,
        result: Line<N>,
    ) {
        let (sum, count) = accumulator;

        // Only count valid positions when count_include_pad=false
        if comptime![!config.count_include_pad] {
            *count += 1;
        }

        *sum += result;
    }

    fn count_position(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        id: u32,
        ih: u32,
        iw: u32,
    ) {
        // When count_include_pad=true, count positions within padded bounds
        // (excludes ceil_mode extensions beyond the padded input)
        if comptime![config.count_include_pad]
            && id < config.padded_d
            && ih < config.padded_h
            && iw < config.padded_w
        {
            let (_sum, count) = accumulator;
            *count += 1;
        }
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: u32,
        output: &mut Tensor<Line<N>>,
        _output_indices: &mut (),
        accumulator: Self::Accumulator,
    ) {
        let (sum, count) = accumulator;
        output[position] = sum / Line::cast_from(count);
    }
}

pub(crate) fn avg_pool3d<R: CubeRuntime>(
    x: CubeTensor<R>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    ceil_mode: bool,
) -> CubeTensor<R> {
    let [batch_size, channels, in_d, in_h, in_w] = x.shape.dims();
    let dilation = 1;

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation,
        in_d,
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation,
        in_h,
        ceil_mode,
    );
    let size_2 = calculate_pool_output_size(
        kernel_size[2],
        stride[2],
        padding[2],
        dilation,
        in_w,
        ceil_mode,
    );

    // Padded dimensions (for count_include_pad with ceil_mode)
    let padded_0 = in_d + 2 * padding[0];
    let padded_1 = in_h + 2 * padding[1];
    let padded_2 = in_w + 2 * padding[2];

    let x = into_contiguous(permute_ncdhw_to_ndhwc(x));
    let line_size = max_line_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, size_2, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, x.dtype);

    let working_units = output.shape.num_elements() / line_size as usize;
    let cube_dim = CubeDim::new(&x.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&x.client, working_units, cube_dim);

    pool3d_direct::launch::<AvgPool3dStrategy, R>(
        &x.client,
        cube_count,
        cube_dim,
        x.as_tensor_arg(line_size),
        output.as_tensor_arg(line_size),
        (),
        Pool3dDirectArgsLaunch::new(
            ScalarArg::new(stride[0] as u32),
            ScalarArg::new(stride[1] as u32),
            ScalarArg::new(stride[2] as u32),
            ScalarArg::new(dilation as u32),
            ScalarArg::new(dilation as u32),
            ScalarArg::new(dilation as u32),
            ScalarArg::new(padding[0] as u32),
            ScalarArg::new(padding[1] as u32),
            ScalarArg::new(padding[2] as u32),
        ),
        (
            kernel_size[0] as u32,
            kernel_size[1] as u32,
            kernel_size[2] as u32,
        ),
        AvgPool3dStrategyConfig {
            count_include_pad,
            padded_d: padded_0 as u32,
            padded_h: padded_1 as u32,
            padded_w: padded_2 as u32,
        },
        output.dtype.into(),
    )
    .expect("Kernel to never fail");

    permute_ndhwc_to_ncdhw(output)
}
