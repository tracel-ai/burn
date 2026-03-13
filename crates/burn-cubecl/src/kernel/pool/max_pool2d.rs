use super::pool2d::{
    Pool2dDirectArgsLaunch, Pool2dDirectStrategy, Pool2dDirectStrategyFamily, pool2d_direct,
};
use crate::{
    CubeRuntime,
    kernel::{
        into_contiguous_aligned,
        pool::pool2d::{Position, view4d},
        utils::{address_type, shape_divmod},
    },
    ops::{
        max_vector_size, numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw,
    },
    tensor::CubeTensor,
};
use burn_backend::{DType, Shape, ops::conv::calculate_pool_output_size};
use cubecl::{
    CubeDim, calculate_cube_count_elemwise, num_traits::Zero, prelude::*, std::tensor::View,
};

struct MaxPoolStrategy;
struct MaxPoolWithIndicesStrategy;

impl Pool2dDirectStrategyFamily for MaxPoolStrategy {
    type Indices<N: Size> = ();
    type Config = ();
    type Pool2d<T: Numeric, N: Size> = Self;
}

impl Pool2dDirectStrategyFamily for MaxPoolWithIndicesStrategy {
    type Indices<N: Size> = View<Vector<i32, N>, Position, ReadWrite>;
    type Config = ();
    type Pool2d<T: Numeric, N: Size> = Self;
}

#[cube]
impl<T: Numeric, N: Size> Pool2dDirectStrategy<T, N> for MaxPoolStrategy {
    type Accumulator = Vector<T, N>;
    type Config = ();
    type Indices = ();

    fn initialize(#[comptime] _config: &Self::Config) -> Self::Accumulator {
        Vector::new(T::min_value())
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        _index: VectorSize,
        result: Vector<T, N>,
    ) {
        *accumulator = max(*accumulator, result);
    }

    fn count_position(
        #[comptime] _config: &Self::Config,
        _accumulator: &mut Self::Accumulator,
        _ih: u32,
        _iw: u32,
    ) {
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: Position,
        output: &mut View<Vector<T, N>, Position, ReadWrite>,
        _output_indices: &mut (),
        accumulator: Self::Accumulator,
    ) {
        output[position] = accumulator;
    }
}

#[cube]
impl<T: Numeric, N: Size> Pool2dDirectStrategy<T, N> for MaxPoolWithIndicesStrategy {
    type Accumulator = (Vector<T, N>, Vector<i32, N>);
    type Config = ();
    type Indices = View<Vector<i32, N>, Position, ReadWrite>;

    fn initialize(#[comptime] _config: &Self::Config) -> Self::Accumulator {
        let val = Vector::new(T::min_value());
        let idx = Vector::zero();
        (val, idx)
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        index: usize,
        result: Vector<T, N>,
    ) {
        let indices = Vector::cast_from(index);
        accumulator.1 = select_many(result.greater_than(accumulator.0), indices, accumulator.1);
        accumulator.0 = max(result, accumulator.0);
    }

    fn count_position(
        #[comptime] _config: &Self::Config,
        _accumulator: &mut Self::Accumulator,
        _ih: u32,
        _iw: u32,
    ) {
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: Position,
        output: &mut View<Vector<T, N>, Position, ReadWrite>,
        output_indices: &mut View<Vector<i32, N>, Position, ReadWrite>,
        accumulator: Self::Accumulator,
    ) {
        output[position] = accumulator.0;
        output_indices[position] = accumulator.1;
    }
}

pub(crate) fn max_pool2d<R: CubeRuntime>(
    x: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> CubeTensor<R> {
    let [batch_size, channels, height, width] = x.meta.shape().dims();

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        height,
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        width,
        ceil_mode,
    );

    let x = into_contiguous_aligned(permute_nchw_to_nhwc(x));

    let vector_size = max_vector_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, x.dtype);

    let working_units = output.meta.num_elements() / vector_size as usize;
    let cube_dim = CubeDim::new(&x.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&x.client, working_units, cube_dim);

    pool2d_direct::launch::<MaxPoolStrategy, R>(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(x, output),
        vector_size,
        x.into_tensor_arg(),
        view4d(output.clone(), vector_size),
        (),
        shape_divmod(&output),
        working_units,
        Pool2dDirectArgsLaunch::new(
            stride[0] as u32,
            stride[1] as u32,
            dilation[0] as u32,
            dilation[1] as u32,
            padding[0] as u32,
            padding[1] as u32,
        ),
        (kernel_size[0] as u32, kernel_size[1] as u32),
        (),
        output.dtype.into(),
    );

    permute_nhwc_to_nchw(output)
}

pub(crate) fn max_pool2d_with_indices<R: CubeRuntime>(
    x: CubeTensor<R>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    dtype_indices: DType,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let [batch_size, channels, size_0, size_1] = x.meta.shape().dims();

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        size_0,
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        size_1,
        ceil_mode,
    );

    let x = into_contiguous_aligned(permute_nchw_to_nhwc(x));
    let vector_size = max_vector_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, channels]);
    let output = empty_device_dtype(
        x.client.clone(),
        x.device.clone(),
        shape_out.clone(),
        x.dtype,
    );
    let indices = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, dtype_indices);

    let working_units = output.meta.num_elements() / vector_size as usize;
    let cube_dim = CubeDim::new(&x.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&x.client, working_units, cube_dim);

    pool2d_direct::launch::<MaxPoolWithIndicesStrategy, R>(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(x, output, indices),
        vector_size,
        x.into_tensor_arg(),
        view4d(output.clone(), vector_size),
        view4d(indices.clone(), vector_size),
        shape_divmod(&output),
        working_units,
        Pool2dDirectArgsLaunch::new(
            stride[0] as u32,
            stride[1] as u32,
            dilation[0] as u32,
            dilation[1] as u32,
            padding[0] as u32,
            padding[1] as u32,
        ),
        (kernel_size[0] as u32, kernel_size[1] as u32),
        (),
        output.dtype.into(),
    );

    let output = permute_nhwc_to_nchw(output);
    let indices = permute_nhwc_to_nchw(indices);
    (output, indices)
}
