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
use cubecl::{CubeDim, calculate_cube_count_elemwise, prelude::*};

struct MaxPoolStrategy;
struct MaxPoolWithIndicesStrategy;

impl Pool2dDirectStrategyFamily for MaxPoolStrategy {
    type Indices = ();
    type Config = ();
    type Pool2d<N: Numeric> = Self;
}

impl Pool2dDirectStrategyFamily for MaxPoolWithIndicesStrategy {
    type Indices = Tensor<Line<i32>>;
    type Config = ();
    type Pool2d<N: Numeric> = Self;
}

#[cube]
impl<N: Numeric> Pool2dDirectStrategy<N> for MaxPoolStrategy {
    type Accumulator = Line<N>;
    type Config = ();
    type Indices = ();

    fn initialize(
        #[comptime] _config: &Self::Config,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        Line::empty(line_size).fill(N::min_value())
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        _index: u32,
        result: Line<N>,
    ) {
        *accumulator = Max::max(*accumulator, result);
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: u32,
        output: &mut Tensor<Line<N>>,
        _output_indices: &mut (),
        accumulator: Self::Accumulator,
    ) {
        output[position] = accumulator;
    }
}

#[cube]
impl<N: Numeric> Pool2dDirectStrategy<N> for MaxPoolWithIndicesStrategy {
    type Accumulator = (Line<N>, Line<i32>);
    type Config = ();
    type Indices = Tensor<Line<i32>>;

    fn initialize(
        #[comptime] _config: &Self::Config,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        let val = Line::empty(line_size).fill(N::min_value());
        let idx = Line::empty(line_size).fill(0i32);
        (val, idx)
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        index: u32,
        result: Line<N>,
    ) {
        let indices = Line::cast_from(index);
        accumulator.1 = select_many(result.greater_than(accumulator.0), indices, accumulator.1);
        accumulator.0 = Max::max(result, accumulator.0);
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: u32,
        output: &mut Tensor<Line<N>>,
        output_indices: &mut Tensor<Line<i32>>,
        accumulator: Self::Accumulator,
    ) {
        output[position] = accumulator.0;
        output_indices[position] = accumulator.1;
    }
}

pub(crate) fn max_pool2d<R: CubeRuntime, E: CubeElement>(
    x: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> CubeTensor<R> {
    let [batch_size, channels, _, _] = x.shape.dims();

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        x.shape[2],
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        x.shape[3],
    );

    let x = into_contiguous(permute_nchw_to_nhwc(x));

    let line_size = max_line_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device::<R, E>(x.client.clone(), x.device.clone(), shape_out);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(output.shape.num_elements() / line_size as usize, cube_dim);

    pool2d_direct::launch::<E, MaxPoolStrategy, R>(
        &x.client,
        cube_count,
        cube_dim,
        x.as_tensor_arg::<E>(line_size),
        output.as_tensor_arg::<E>(line_size),
        (),
        Pool2dDirectArgsLaunch::new(
            ScalarArg::new(stride[0] as u32),
            ScalarArg::new(stride[1] as u32),
            ScalarArg::new(dilation[0] as u32),
            ScalarArg::new(dilation[1] as u32),
            ScalarArg::new(padding[0] as u32),
            ScalarArg::new(padding[1] as u32),
        ),
        (kernel_size[0] as u32, kernel_size[1] as u32),
        (),
    );

    permute_nhwc_to_nchw(output)
}

pub(crate) fn max_pool2d_with_indices<R: CubeRuntime, E: CubeElement, I: CubeElement>(
    x: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> (CubeTensor<R>, CubeTensor<R>) {
    let [batch_size, channels, _, _] = x.shape.dims();

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        x.shape[2],
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        x.shape[3],
    );

    let x = into_contiguous(permute_nchw_to_nhwc(x));
    let line_size = max_line_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device::<R, E>(x.client.clone(), x.device.clone(), shape_out.clone());
    let indices = empty_device::<R, I>(x.client.clone(), x.device.clone(), shape_out);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(output.shape.num_elements() / line_size as usize, cube_dim);

    pool2d_direct::launch::<E, MaxPoolWithIndicesStrategy, R>(
        &x.client,
        cube_count,
        cube_dim,
        x.as_tensor_arg::<E>(line_size),
        output.as_tensor_arg::<E>(line_size),
        indices.as_tensor_arg::<I>(line_size),
        Pool2dDirectArgsLaunch::new(
            ScalarArg::new(stride[0] as u32),
            ScalarArg::new(stride[1] as u32),
            ScalarArg::new(dilation[0] as u32),
            ScalarArg::new(dilation[1] as u32),
            ScalarArg::new(padding[0] as u32),
            ScalarArg::new(padding[1] as u32),
        ),
        (kernel_size[0] as u32, kernel_size[1] as u32),
        (),
    );

    let output = permute_nhwc_to_nchw(output);
    let indices = permute_nhwc_to_nchw(indices);
    (output, indices)
}
