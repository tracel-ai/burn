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
use burn_backend::{DType, Shape, ops::conv::calculate_pool_output_size};
use cubecl::{CubeDim, calculate_cube_count_elemwise, prelude::*};

struct MaxPool3dStrategy;
struct MaxPool3dWithIndicesStrategy;

impl Pool3dDirectStrategyFamily for MaxPool3dStrategy {
    type Indices = ();
    type Config = ();
    type Pool3d<N: Numeric> = Self;
}

impl Pool3dDirectStrategyFamily for MaxPool3dWithIndicesStrategy {
    type Indices = Tensor<Line<i32>>;
    type Config = ();
    type Pool3d<N: Numeric> = Self;
}

#[cube]
impl<N: Numeric> Pool3dDirectStrategy<N> for MaxPool3dStrategy {
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

    fn count_position(
        #[comptime] _config: &Self::Config,
        _accumulator: &mut Self::Accumulator,
        _id: u32,
        _ih: u32,
        _iw: u32,
    ) {
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
impl<N: Numeric> Pool3dDirectStrategy<N> for MaxPool3dWithIndicesStrategy {
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

    fn count_position(
        #[comptime] _config: &Self::Config,
        _accumulator: &mut Self::Accumulator,
        _id: u32,
        _ih: u32,
        _iw: u32,
    ) {
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

pub(crate) fn max_pool3d<R: CubeRuntime>(
    x: CubeTensor<R>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> CubeTensor<R> {
    let [batch_size, channels, _, _, _] = x.shape.dims();

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        x.shape[2],
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        x.shape[3],
        ceil_mode,
    );
    let size_2 = calculate_pool_output_size(
        kernel_size[2],
        stride[2],
        padding[2],
        dilation[2],
        x.shape[4],
        ceil_mode,
    );

    let x = into_contiguous(permute_ncdhw_to_ndhwc(x));

    let line_size = max_line_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, size_2, channels]);
    let output = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, x.dtype);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(output.shape.num_elements() / line_size as usize, cube_dim);

    pool3d_direct::launch::<MaxPool3dStrategy, R>(
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
            ScalarArg::new(dilation[0] as u32),
            ScalarArg::new(dilation[1] as u32),
            ScalarArg::new(dilation[2] as u32),
            ScalarArg::new(padding[0] as u32),
            ScalarArg::new(padding[1] as u32),
            ScalarArg::new(padding[2] as u32),
        ),
        (
            kernel_size[0] as u32,
            kernel_size[1] as u32,
            kernel_size[2] as u32,
        ),
        (),
        output.dtype.into(),
    )
    .expect("Kernel to never fail");

    permute_ndhwc_to_ncdhw(output)
}

pub(crate) fn max_pool3d_with_indices<R: CubeRuntime>(
    x: CubeTensor<R>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
    dtype_indices: DType,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let [batch_size, channels, _, _, _] = x.shape.dims();

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        x.shape[2],
        ceil_mode,
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        x.shape[3],
        ceil_mode,
    );
    let size_2 = calculate_pool_output_size(
        kernel_size[2],
        stride[2],
        padding[2],
        dilation[2],
        x.shape[4],
        ceil_mode,
    );

    let x = into_contiguous(permute_ncdhw_to_ndhwc(x));
    let line_size = max_line_size(&x);

    let shape_out = Shape::new([batch_size, size_0, size_1, size_2, channels]);
    let output = empty_device_dtype(
        x.client.clone(),
        x.device.clone(),
        shape_out.clone(),
        x.dtype,
    );
    let indices = empty_device_dtype(x.client.clone(), x.device.clone(), shape_out, dtype_indices);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(output.shape.num_elements() / line_size as usize, cube_dim);

    pool3d_direct::launch::<MaxPool3dWithIndicesStrategy, R>(
        &x.client,
        cube_count,
        cube_dim,
        x.as_tensor_arg(line_size),
        output.as_tensor_arg(line_size),
        indices.as_tensor_arg(line_size),
        Pool3dDirectArgsLaunch::new(
            ScalarArg::new(stride[0] as u32),
            ScalarArg::new(stride[1] as u32),
            ScalarArg::new(stride[2] as u32),
            ScalarArg::new(dilation[0] as u32),
            ScalarArg::new(dilation[1] as u32),
            ScalarArg::new(dilation[2] as u32),
            ScalarArg::new(padding[0] as u32),
            ScalarArg::new(padding[1] as u32),
            ScalarArg::new(padding[2] as u32),
        ),
        (
            kernel_size[0] as u32,
            kernel_size[1] as u32,
            kernel_size[2] as u32,
        ),
        (),
        output.dtype.into(),
    )
    .expect("Kernel to never fail");

    let output = permute_ndhwc_to_ncdhw(output);
    let indices = permute_ndhwc_to_ncdhw(indices);
    (output, indices)
}
