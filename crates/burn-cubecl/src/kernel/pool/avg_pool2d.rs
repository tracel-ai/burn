use super::pool2d::{
    Pool2dDirectArgsLaunch, Pool2dDirectStrategy, Pool2dDirectStrategyFamily, pool2d_direct,
};
use crate::{
    CubeRuntime,
    element::CubeElement,
    kernel::into_contiguous,
    ops::{max_line_size, numeric::empty_device, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_tensor::{Shape, ops::conv::calculate_pool_output_size};
use cubecl::prelude::*;
use cubecl::{CubeDim, calculate_cube_count_elemwise, prelude::ScalarArg};

struct AvgPoolStrategy;

impl Pool2dDirectStrategyFamily for AvgPoolStrategy {
    type Indices = ();
    type Config = AvgPoolStrategyConfig;
    type Pool2d<N: Numeric> = Self;
}

#[derive(CubeType, Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct AvgPoolStrategyConfig {
    kernel_size_h: u32,
    kernel_size_w: u32,
    count_include_pad: bool,
}

#[cube]
impl<N: Numeric> Pool2dDirectStrategy<N> for AvgPoolStrategy {
    type Accumulator = (Line<N>, u32);
    type Config = AvgPoolStrategyConfig;
    type Indices = ();

    fn initialize(
        #[comptime] config: &Self::Config,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        let sum = Line::empty(line_size).fill(N::from_int(0));
        let count = comptime! {if config.count_include_pad {
            config.kernel_size_h * config.kernel_size_w
        } else {
            0u32
        }};

        (sum, count)
    }

    fn accumulate(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        _index: u32,
        result: Line<N>,
    ) {
        let (sum, count) = accumulator;

        if comptime![!config.count_include_pad] {
            *count += 1;
        }

        *sum += result;
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

pub(crate) fn avg_pool2d<R: CubeRuntime, E: CubeElement>(
    x: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> CubeTensor<R> {
    let [batch_size, channels, _, _] = x.shape.dims();
    let dilation = 1;

    let size_0 =
        calculate_pool_output_size(kernel_size[0], stride[0], padding[0], dilation, x.shape[2]);
    let size_1 =
        calculate_pool_output_size(kernel_size[1], stride[1], padding[1], dilation, x.shape[3]);

    let x = into_contiguous(permute_nchw_to_nhwc(x));
    let line_size = max_line_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device::<R, E>(x.client.clone(), x.device.clone(), shape_out);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(output.shape.num_elements() / line_size as usize, cube_dim);

    pool2d_direct::launch::<E, AvgPoolStrategy, R>(
        &x.client,
        cube_count,
        cube_dim,
        x.as_tensor_arg(line_size),
        output.as_tensor_arg(line_size),
        (),
        Pool2dDirectArgsLaunch::new(
            ScalarArg::new(stride[0] as u32),
            ScalarArg::new(stride[1] as u32),
            ScalarArg::new(dilation as u32),
            ScalarArg::new(dilation as u32),
            ScalarArg::new(padding[0] as u32),
            ScalarArg::new(padding[1] as u32),
        ),
        (kernel_size[0] as u32, kernel_size[1] as u32),
        AvgPoolStrategyConfig {
            kernel_size_h: kernel_size[0] as u32,
            kernel_size_w: kernel_size[1] as u32,
            count_include_pad,
        },
    );

    permute_nhwc_to_nchw(output)
}
