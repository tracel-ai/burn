use super::pool2d::{
    pool2d_direct, Pool2dDirectArgsLaunch, Pool2dDirectStrategy, Pool2dDirectStrategyFamily,
};
use crate::{element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor, CubeRuntime};
use burn_tensor::{ops::conv::calculate_pool_output_size, Shape};
use cubecl::{calculate_cube_count_elemwise, prelude::*, CubeDim};

struct MaxPoolStrategy;
struct MaxPoolWithIndicesStrategy;

impl Pool2dDirectStrategyFamily for MaxPoolStrategy {
    type Indices = ();
    type Config = ();
    type Pool2d<N: Numeric> = Self;
}

impl Pool2dDirectStrategyFamily for MaxPoolWithIndicesStrategy {
    type Indices = Tensor<i32>;
    type Config = ();
    type Pool2d<N: Numeric> = Self;
}

#[cube]
impl<N: Numeric> Pool2dDirectStrategy<N> for MaxPoolStrategy {
    type Accumulator = N;
    type Config = ();
    type Indices = ();

    fn initialize(#[comptime] _config: &Self::Config) -> Self::Accumulator {
        N::min_value()
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        _index: u32,
        result: N,
    ) {
        if result > *accumulator {
            *accumulator = result;
        }
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: u32,
        output: &mut Tensor<N>,
        _output_indices: &mut (),
        accumulator: Self::Accumulator,
    ) {
        output[position] = accumulator;
    }
}

#[cube]
impl<N: Numeric> Pool2dDirectStrategy<N> for MaxPoolWithIndicesStrategy {
    type Accumulator = (N, i32);
    type Config = ();
    type Indices = Tensor<i32>;

    fn initialize(#[comptime] _config: &Self::Config) -> Self::Accumulator {
        (N::min_value(), 0i32)
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        index: u32,
        result: N,
    ) {
        if result > accumulator.0 {
            accumulator.0 = result;
            accumulator.1 = i32::cast_from(index);
        }
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: u32,
        output: &mut Tensor<N>,
        output_indices: &mut Tensor<i32>,
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
        x.shape.dims[2],
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        x.shape.dims[3],
    );

    let shape_out = Shape::new([batch_size, channels, size_0, size_1]);
    let output = empty_device::<R, E>(x.client.clone(), x.device.clone(), shape_out);

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    pool2d_direct::launch::<E, MaxPoolStrategy, R>(
        &x.client,
        cube_count,
        cube_dim,
        x.as_tensor_arg::<E>(1),
        output.as_tensor_arg::<E>(1),
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

    output
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
        x.shape.dims[2],
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        x.shape.dims[3],
    );

    let shape_out = Shape::new([batch_size, channels, size_0, size_1]);
    let output = empty_device::<R, E>(x.client.clone(), x.device.clone(), shape_out.clone());
    let indices = empty_device::<R, I>(x.client.clone(), x.device.clone(), shape_out);

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    pool2d_direct::launch::<E, MaxPoolWithIndicesStrategy, R>(
        &x.client,
        cube_count,
        cube_dim,
        x.as_tensor_arg::<E>(1),
        output.as_tensor_arg::<E>(1),
        indices.as_tensor_arg::<I>(1),
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
    (output, indices)
}
