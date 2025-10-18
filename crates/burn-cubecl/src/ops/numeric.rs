use crate::{CubeRuntime, FloatElement, IntElement, kernel::utils::linear_view};
use crate::{element::CubeElement, tensor::CubeTensor};
use crate::{
    kernel::{
        AddOp, BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, DivOp, MulOp, PowOp, RemainderOp, SubOp,
        launch_binop, launch_binop_int, launch_scalar_binop, launch_scalar_binop_int,
    },
    ops::max_line_size,
};
use burn_tensor::{ElementConversion, Shape};
use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use cubecl::{client::ComputeClient, server::Allocation};

/// Create a tensor filled with `value`
pub fn full<R: CubeRuntime, E: CubeElement>(
    shape: Shape,
    device: &R::Device,
    value: E,
) -> CubeTensor<R> {
    let client = R::client(device);

    full_device::<R, E>(client, shape, device.clone(), value)
}

/// Create a tensor filled with `value`
pub fn full_device<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server>,
    shape: Shape,
    device: R::Device,
    value: E,
) -> CubeTensor<R> {
    let empty = empty_device::<R, E>(client, device, shape);

    #[cube(launch)]
    pub fn full_kernel<C: Numeric>(tensor: &mut LinearView<C, ReadWrite>, value: C) {
        if !tensor.is_in_bounds(ABSOLUTE_POS) {
            terminate!();
        }

        tensor[ABSOLUTE_POS] = value;
    }

    let num_elems = empty.shape.num_elements();
    let line_size = max_line_size(&empty);

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    full_kernel::launch::<E, R>(
        &empty.client,
        cube_count,
        cube_dim,
        linear_view(&empty, line_size),
        ScalarArg::new(value),
    );

    empty
}

/// Create a tensor filled with zeros
pub fn zeros<R: CubeRuntime, E: CubeElement>(shape: Shape, device: &R::Device) -> CubeTensor<R> {
    let client = R::client(device);

    zeros_device::<R, E>(client, device.clone(), shape)
}

/// Create a tensor filled with zeros
pub fn zeros_device<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    full_device::<R, E>(client, shape, device, 0.elem())
}

/// Create a tensor filled with ones
pub fn ones<R: CubeRuntime, E: CubeElement>(shape: Shape, device: &R::Device) -> CubeTensor<R> {
    let client = R::client(device);

    ones_device::<R, E>(client, device.clone(), shape)
}

/// Create a tensor filled with ones
pub fn ones_device<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    full_device::<R, E>(client, shape, device, 1.elem())
}

/// Create a tensor with uninitialized memory
pub fn empty_device<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    CubeTensor::new_contiguous(client, device, shape, buffer, E::dtype())
}

/// Create a tensor with uninitialized memory
pub fn empty_device_strided<R: CubeRuntime, E: CubeElement>(
    client: ComputeClient<R::Server>,
    device: R::Device,
    shape: Shape,
) -> CubeTensor<R> {
    let Allocation { handle, strides } = client.empty_tensor(&shape.dims, size_of::<E>());

    CubeTensor::new(client, handle, shape, device, strides, E::dtype())
}

/// Add two tensors
pub fn add<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, AddOp>(lhs, rhs)
}

/// Add a tensor and a scalar
pub fn add_scalar<R: CubeRuntime, E: CubeElement>(lhs: CubeTensor<R>, rhs: E) -> CubeTensor<R> {
    launch_scalar_binop::<R, E, AddOp>(lhs, rhs)
}

/// Subtract two tensors
pub fn sub<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, SubOp>(lhs, rhs)
}

/// Subtract a tensor and a scalar
pub fn sub_scalar<R: CubeRuntime, E: CubeElement>(lhs: CubeTensor<R>, rhs: E) -> CubeTensor<R> {
    launch_scalar_binop::<R, E, SubOp>(lhs, rhs)
}

/// Multiply two tensors
pub fn mul<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, MulOp>(lhs, rhs)
}

/// Multiply a tensor and a scalar
pub fn mul_scalar<R: CubeRuntime, E: CubeElement>(lhs: CubeTensor<R>, rhs: E) -> CubeTensor<R> {
    launch_scalar_binop::<R, E, MulOp>(lhs, rhs)
}

/// Divide two tensors
pub fn div<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, DivOp>(lhs, rhs)
}

/// Divide a tensor by a scalar
pub fn div_scalar<R: CubeRuntime, E: CubeElement>(lhs: CubeTensor<R>, rhs: E) -> CubeTensor<R> {
    launch_scalar_binop::<R, E, DivOp>(lhs, rhs)
}

/// Calculate remainder of two tensors
pub fn remainder<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, RemainderOp>(lhs, rhs)
}

/// Calculate the remainder of a tensor with a scalar
pub fn remainder_scalar<R: CubeRuntime, E: CubeElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_binop::<R, E, RemainderOp>(lhs, rhs)
}

/// Calculate the power of two tensors
pub fn pow<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop::<R, E, PowOp<E>>(lhs, rhs)
}

/// Bitwise and two tensors
pub fn bitwise_and<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop_int::<R, E, BitwiseAndOp>(lhs, rhs)
}

/// Bitwise and with a scalar
pub fn bitwise_and_scalar<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_binop_int::<R, E, BitwiseAndOp>(lhs, rhs)
}

/// Bitwise or two tensors
pub fn bitwise_or<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop_int::<R, E, BitwiseOrOp>(lhs, rhs)
}

/// Bitwise or with a scalar
pub fn bitwise_or_scalar<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_binop_int::<R, E, BitwiseOrOp>(lhs, rhs)
}

/// Bitwise xor two tensors
pub fn bitwise_xor<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    launch_binop_int::<R, E, BitwiseXorOp>(lhs, rhs)
}

/// Bitwise xor with a scalar
pub fn bitwise_xor_scalar<R: CubeRuntime, E: IntElement>(
    lhs: CubeTensor<R>,
    rhs: E,
) -> CubeTensor<R> {
    launch_scalar_binop_int::<R, E, BitwiseXorOp>(lhs, rhs)
}

/// Operation family trait for cumulative operations
pub(crate) trait CumulativeOpFamily: Send + Sync + 'static {
    type CumulativeOp<C: Numeric>: CumulativeOp<C>;
}

/// Trait for cumulative operations
#[cube]
pub(crate) trait CumulativeOp<C: Numeric>: 'static + Send + Sync {
    /// Execute a cumulative operation
    fn execute(lhs: C, rhs: C) -> C;

    /// Get the initial value for the accumulator
    fn init_value(first_element: C) -> C;
}

// Operation types
struct SumOp;
struct ProdOp;
struct MaxOp;
struct MinOp;

// Implement CumulativeOpFamily for each operation
impl CumulativeOpFamily for SumOp {
    type CumulativeOp<C: Numeric> = Self;
}

impl CumulativeOpFamily for ProdOp {
    type CumulativeOp<C: Numeric> = Self;
}

impl CumulativeOpFamily for MaxOp {
    type CumulativeOp<C: Numeric> = Self;
}

impl CumulativeOpFamily for MinOp {
    type CumulativeOp<C: Numeric> = Self;
}

// Implement CumulativeOp for each operation type
#[cube]
impl<N: Numeric> CumulativeOp<N> for SumOp {
    fn execute(lhs: N, rhs: N) -> N {
        lhs + rhs
    }

    fn init_value(_first_element: N) -> N {
        N::from_int(0)
    }
}

#[cube]
impl<N: Numeric> CumulativeOp<N> for ProdOp {
    fn execute(lhs: N, rhs: N) -> N {
        lhs * rhs
    }

    fn init_value(_first_element: N) -> N {
        N::from_int(1)
    }
}

#[cube]
impl<N: Numeric> CumulativeOp<N> for MaxOp {
    fn execute(lhs: N, rhs: N) -> N {
        N::max(lhs, rhs)
    }

    fn init_value(first_element: N) -> N {
        first_element
    }
}

#[cube]
impl<N: Numeric> CumulativeOp<N> for MinOp {
    fn execute(lhs: N, rhs: N) -> N {
        N::min(lhs, rhs)
    }

    fn init_value(first_element: N) -> N {
        first_element
    }
}

/// Generic cumulative operation kernel
///
/// # Limitations
///
/// This is a **naive sequential implementation** along the cumulative dimension:
/// - Each output element sequentially reads all previous elements along the dimension
/// - Computational complexity: O(n^2) memory reads where n is the size of the cumulative dimension
/// - **Performance:** Suitable for small tensors or small dimensions. For large tensors,
///   performance will degrade significantly compared to an optimized parallel scan algorithm.
///
/// # TODO
///
/// Implement an efficient GPU-optimized parallel scan algorithm.
#[cube(launch)]
fn cumulative_kernel<C: Numeric, O: CumulativeOpFamily>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    dim_stride: u32,
    #[comptime] dim_size: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let before_dim = idx / dim_stride;
    let after_dim = idx % dim_stride;
    let dim_offset = (idx / dim_stride) % dim_size;

    // Read first element
    let first_read_idx = (before_dim / dim_size) * (dim_size * dim_stride) + after_dim;
    let first_elem = input[first_read_idx];

    // Initialize accumulator
    let mut result = O::CumulativeOp::<C>::init_value(first_elem);

    // Accumulate values
    for i in 0..dim_size {
        if i <= dim_offset {
            let read_idx =
                (before_dim / dim_size) * (dim_size * dim_stride) + i * dim_stride + after_dim;
            result = O::CumulativeOp::<C>::execute(result, input[read_idx]);
        }
    }
    output[idx] = result;
}

/// Compute the cumulative sum along a dimension
pub fn cumsum<R: CubeRuntime, E: CubeElement>(input: CubeTensor<R>, dim: usize) -> CubeTensor<R> {
    cumulative_op::<R, E, SumOp>(input, dim)
}

/// Compute the cumulative product along a dimension
pub fn cumprod<R: CubeRuntime, E: CubeElement>(input: CubeTensor<R>, dim: usize) -> CubeTensor<R> {
    cumulative_op::<R, E, ProdOp>(input, dim)
}

/// Compute the cumulative minimum along a dimension
pub fn cummin<R: CubeRuntime, E: CubeElement>(input: CubeTensor<R>, dim: usize) -> CubeTensor<R> {
    cumulative_op::<R, E, MinOp>(input, dim)
}

/// Compute the cumulative maximum along a dimension
pub fn cummax<R: CubeRuntime, E: CubeElement>(input: CubeTensor<R>, dim: usize) -> CubeTensor<R> {
    cumulative_op::<R, E, MaxOp>(input, dim)
}

/// Generic cumulative operation function
fn cumulative_op<R: CubeRuntime, E: CubeElement, O: CumulativeOpFamily>(
    input: CubeTensor<R>,
    dim: usize,
) -> CubeTensor<R> {
    let client = input.client.clone();
    let device = input.device.clone();
    let shape = input.shape.clone();
    let dim_size = shape.dims[dim];

    let dim_stride: usize = shape.dims[dim + 1..].iter().product();
    let output = empty_device::<R, E>(client.clone(), device, shape);

    let num_elems = output.shape.num_elements();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    cumulative_kernel::launch::<E, O, R>(
        &client,
        cube_count,
        cube_dim,
        unsafe {
            TensorArg::from_raw_parts::<E>(&input.handle, &input.strides, &input.shape.dims, 1)
        },
        unsafe {
            TensorArg::from_raw_parts::<E>(&output.handle, &output.strides, &output.shape.dims, 1)
        },
        ScalarArg::new(dim_stride as u32),
        dim_size as u32,
    );

    output
}
